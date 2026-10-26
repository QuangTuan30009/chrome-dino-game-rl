import argparse
import csv
from collections import deque, namedtuple
import datetime
from itertools import count
import os
import shutil
import gymnasium as gym
import torch
import torch.nn as nn
import random
import envs
import numpy as np
import matplotlib.pyplot as plt

from model import DQN, create_model

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Thử import TensorBoard nếu có cài đặt
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)


class MemoryReplay(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(
        self,
        env: envs.Wrapper,
        policy_net: nn.Module,
        target_net: nn.Module,
        n_episodes=1_000,
        lr=1e-4,
        batch_size=32,
        replay_size=10_000,  # experience replay's buffer size
        learning_start=10_000,  # number of frames before learning starts
        target_update_freq=1_000,  # number of frames between every target network update
        optimize_freq=1,
        gamma=0.99,  # reward decay factor
        # explore/exploit eps-greedy policy
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=10_000,
        algo="double_dqn",  # 'dqn' hoặc 'double_dqn'
    ):
        self.env = env

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        print(f"--> Sử dụng thiết bị: {self.device}")
        print(f"--> Thuật toán: {algo.upper()}")

        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory_replay = MemoryReplay(replay_size)

        self.n_steps = 0
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.algo = algo

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        self.learning_start = learning_start
        self.target_update_freq = target_update_freq
        self.optimize_freq = optimize_freq

        self.gamma = gamma

        self._get_eps = lambda n_steps: eps_end + (eps_start - eps_end) * np.exp(
            -1.0 * n_steps / eps_decay
        )

        # Khởi tạo thư mục lưu kết quả
        folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        folder_path = os.path.join("results", folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        self.folder_path = folder_path

        # Metrics logger
        self.metrics_path = os.path.join(folder_path, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps", "epsilon", "loss"])

        self._loss_buffer: list[float] = []   # tích lũy loss trong episode
        self._reward_history: list[float] = []  # lịch sử reward để theo dõi best
        self._best_avg_reward = float("-inf")

        # TensorBoard writer (nếu có)
        if _TENSORBOARD_AVAILABLE:
            self._tb_writer = SummaryWriter(log_dir=folder_path)
            print(f"--> TensorBoard log tại: {folder_path}")
        else:
            self._tb_writer = None

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select the next action given the current state following the eps-greedy policy"""
        eps = self._get_eps(self.n_steps)

        if random.random() > eps:
            # exploit
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).max(dim=1)[1][0]
        else:
            # explore
            return torch.tensor(self.env.action_space.sample(), device=self.device)

    def _optimize(self) -> float | None:
        """Thực hiện một bước tối ưu hóa. Trả về giá trị loss."""
        if len(self.memory_replay) < self.batch_size:
            return None

        transitions = self.memory_replay.sample(self.batch_size)

        # Convert batch-array of Transitions to a Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.stack(batch.reward)
        terminated_batch = torch.tensor(
            batch.terminated, device=self.device, dtype=torch.float
        )

        # Compute batch "Q(s, a)"
        # The model returns "Q(s)", then we select the columns of actions taken.
        Q_values = (
            self.policy_net(state_batch)
            .gather(1, action_batch.unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            if self.algo == "double_dqn":
                # === Double DQN ===
                # Bước 1: policy_net chọn hành động tốt nhất ở trạng thái kế tiếp
                next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(-1)
                # Bước 2: target_net đánh giá giá trị của hành động đó
                # => Tách biệt "chọn action" và "đánh giá action" để loại bỏ overestimation bias
                next_Q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(-1)
            else:
                # === DQN gốc ===
                next_Q_values = self.target_net(next_state_batch).max(1)[0]

        expected_Q_values = (
            1.0 - terminated_batch
        ) * next_Q_values * self.gamma + reward_batch

        # Compute Huber loss (ổn định hơn MSE với outliers)
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, expected_Q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def _log_episode(self, episode_i: int, total_reward: float, steps: int):
        """Ghi log metrics sau mỗi episode."""
        eps = self._get_eps(self.n_steps)
        avg_loss = float(np.mean(self._loss_buffer)) if self._loss_buffer else 0.0
        self._loss_buffer.clear()

        # Ghi ra CSV
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_i, round(total_reward, 4), steps, round(eps, 4), round(avg_loss, 6)])

        # Ghi ra TensorBoard nếu có
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("Episode/Reward", total_reward, episode_i)
            self._tb_writer.add_scalar("Episode/Steps", steps, episode_i)
            self._tb_writer.add_scalar("Episode/Epsilon", eps, episode_i)
            self._tb_writer.add_scalar("Episode/Loss", avg_loss, episode_i)

        self._reward_history.append(total_reward)

        # Kiểm tra và lưu best model
        if len(self._reward_history) >= 10:
            recent_avg = float(np.mean(self._reward_history[-10:]))
            if recent_avg > self._best_avg_reward:
                self._best_avg_reward = recent_avg
                best_path = os.path.join(self.folder_path, "best_model.pth")
                torch.save(self.policy_net.state_dict(), best_path)
                print(f"  --> [Best Model] Episode {episode_i}: avg reward (10 eps) = {recent_avg:.2f}")

    def _save_learning_curve(self):
        """Vẽ và lưu biểu đồ learning curve sau khi train xong."""
        if not os.path.exists(self.metrics_path):
            return

        episodes, rewards, steps_list, epsilons, losses = [], [], [], [], []
        with open(self.metrics_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                steps_list.append(int(row["steps"]))
                epsilons.append(float(row["epsilon"]))
                losses.append(float(row["loss"]))

        if len(episodes) < 2:
            return

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Learning Curve - {self.algo.upper()}", fontsize=14)

        # Reward
        axs[0, 0].plot(episodes, rewards, alpha=0.4, color="steelblue", label="Raw")
        if len(rewards) >= 10:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode="valid")
            axs[0, 0].plot(episodes[9:], smoothed, color="blue", linewidth=2, label="Smoothed(10)")
        axs[0, 0].set_title("Episode Reward")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")
        axs[0, 0].legend()

        # Episode Length
        axs[0, 1].plot(episodes, steps_list, alpha=0.4, color="green", label="Raw")
        if len(steps_list) >= 10:
            smoothed_steps = np.convolve(steps_list, np.ones(10)/10, mode="valid")
            axs[0, 1].plot(episodes[9:], smoothed_steps, color="darkgreen", linewidth=2, label="Smoothed(10)")
        axs[0, 1].set_title("Episode Length (Steps)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Steps")
        axs[0, 1].legend()

        # Loss
        axs[1, 0].plot(episodes, losses, alpha=0.6, color="red")
        axs[1, 0].set_title("Average Loss Per Episode")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Loss")

        # Epsilon decay
        axs[1, 1].plot(episodes, epsilons, color="orange", linewidth=2)
        axs[1, 1].set_title("Epsilon Decay (Exploration Rate)")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Epsilon")

        plt.tight_layout()
        curve_path = os.path.join(self.folder_path, "learning_curve.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"--> Đã lưu learning curve tại: {curve_path}")

    def train(self):
        for episode_i in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)

            total_reward = 0.0
            episode_steps = 0

            for t in count():
                self.n_steps += 1
                episode_steps += 1

                action = self._select_action(state)

                next_state, reward, terminated, truncated, *_ = self.env.step(
                    action.item()
                )
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)

                total_reward += float(reward)

                self.memory_replay.push(
                    state,
                    action,
                    next_state,
                    torch.tensor(reward, device=self.device, dtype=torch.float32),
                    terminated or truncated,
                )

                # Synchronize the target network with the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.target_update_freq == 0
                ):
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Optimize the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.optimize_freq == 0
                ):
                    loss_val = self._optimize()
                    if loss_val is not None:
                        self._loss_buffer.append(loss_val)

                if terminated or truncated:
                    print(
                        f"Episode {episode_i:4d} | steps: {episode_steps:5d} | reward: {total_reward:8.2f} | eps: {self._get_eps(self.n_steps):.3f}"
                    )
                    self._log_episode(episode_i, total_reward, episode_steps)
                    break
                else:
                    state = next_state

            # Lưu checkpoint định kỳ (state_dict)
            if episode_i % 50 == 0:
                self.save_obs_result(episode_i, self.env.frames)
                self.save_model_weights(episode_i)

        # Sau khi train xong: vẽ learning curve
        self._save_learning_curve()

        if self._tb_writer is not None:
            self._tb_writer.close()

        self.env.close()

    def save_obs_result(self, episode_i: int, obs_arr: list[np.ndarray]):
        from PIL import Image
        if not obs_arr:
            return
        frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
        file_path = os.path.join(self.folder_path, f"episode-{episode_i}.gif")

        frames[0].save(
            file_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def save_model_weights(self, episode_i: int):
        """Lưu checkpoint theo state_dict (chuẩn PyTorch)."""
        file_path = os.path.join(self.folder_path, f"model-{episode_i}.pth")
        torch.save(self.policy_net.state_dict(), file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on Chrome Dino Game")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Đường dẫn tới checkpoint model để tiếp tục/fine-tune")
    parser.add_argument("-e", "--episodes", type=int, default=1000,
                        help="Số episodes huấn luyện (mặc định: 1000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (mặc định: 1e-4)")
    parser.add_argument("--eps_start", type=float, default=None,
                        help="Epsilon khởi đầu (mặc định: 1.0 mới / 0.15 fine-tune)")
    parser.add_argument("--learning_start", type=int, default=None,
                        help="Số bước trước khi bắt đầu optimize (mặc định: 10000 / 1000 fine-tune)")
    parser.add_argument("--algo", type=str, default="double_dqn",
                        choices=["dqn", "double_dqn"],
                        help="Thuật toán cập nhật Q-value: 'dqn' hoặc 'double_dqn' (mặc định: double_dqn)")
    parser.add_argument("--model_type", type=str, default="dueling",
                        choices=["dqn", "dueling"],
                        help="Kiến trúc mạng: 'dqn' hoặc 'dueling' (mặc định: dueling)")
    parser.add_argument("--survival_reward", type=float, default=0.01,
                        help="Phần thưởng mỗi frame sống sót (mặc định: 0.01)")
    parser.add_argument("--collision_penalty", type=float, default=-5.0,
                        help="Hình phạt khi va chạm (mặc định: -5.0)")

    args = parser.parse_args()

    env = gym.make(
        "Env-v0",
        render_mode="rgb_array",
        game_mode="train",
        survival_reward=args.survival_reward,
        collision_penalty=args.collision_penalty,
        terminate_on_collision=True,
    )
    env = envs.Wrapper(env, k=4)

    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n  # type: ignore

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if args.model_path and os.path.exists(args.model_path):
        print(f"--> Tiếp tục huấn luyện (Fine-tuning) từ model: {args.model_path}")
        # Hỗ trợ cả state_dict (mới) lẫn full model object (cũ)
        checkpoint = torch.load(args.model_path, weights_only=False, map_location=device)
        policy_net = create_model(args.model_type, in_channels, out_channels)
        if isinstance(checkpoint, dict):
            # state_dict format (mới)
            policy_net.load_state_dict(checkpoint)
        else:
            # full object format (cũ) - sao chép trọng số sang mô hình mới
            try:
                policy_net.load_state_dict(checkpoint.state_dict())
                print("  --> Đã chuyển đổi từ full model sang state_dict thành công.")
            except Exception:
                policy_net = checkpoint
                print("  --> Sử dụng full model object cũ trực tiếp.")
        target_net = create_model(args.model_type, in_channels, out_channels)
        target_net.load_state_dict(policy_net.state_dict())
        eps_start = args.eps_start if args.eps_start is not None else 0.15
        learning_start = args.learning_start if args.learning_start is not None else 1000
    else:
        print(f"--> Bắt đầu huấn luyện mới [{args.model_type.upper()} + {args.algo.upper()}]...")
        policy_net = create_model(args.model_type, in_channels, out_channels)
        target_net = create_model(args.model_type, in_channels, out_channels)
        eps_start = args.eps_start if args.eps_start is not None else 1.0
        learning_start = args.learning_start if args.learning_start is not None else 10000

    trainer = Trainer(
        env,
        policy_net,
        target_net,
        n_episodes=args.episodes,
        lr=args.lr,
        eps_start=eps_start,
        learning_start=learning_start,
        algo=args.algo,
    )
    trainer.train()
