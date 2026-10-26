
import gymnasium as gym
import pygame
import torch
import envs
import argparse
import matplotlib.pyplot as plt
import numpy as np

from model import DQN, create_model


def human_play(show_plot: bool = False):
    env = gym.make("Env-v0", render_mode="human", max_episode_steps=1000)
    obs, _ = env.reset()

    total_reward = 0.0
    n_frames = 0
    while True:
        n_frames += 1
        userInput = pygame.key.get_pressed()
        action = envs.Action.STAND
        if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
            action = envs.Action.JUMP
        elif userInput[pygame.K_DOWN]:
            action = envs.Action.DUCK

        obs, reward, terminated, _, _ = env.step(action)

        total_reward += float(reward)
        if terminated:
            break

    print(f"Total reward: {total_reward}, number of frames: {n_frames}")

    env.close()

    if show_plot:
        plt.imshow(obs)
        plt.show()


def load_model(model_path: str, model_type: str, in_channels: int, out_channels: int, device: torch.device):
    """Tải model linh hoạt: hỗ trợ cả state_dict (mới) lẫn full object (cũ).

    Args:
        model_path: Đường dẫn file .pth.
        model_type: Kiến trúc mạng ('dqn' hoặc 'dueling').
        in_channels: Số kênh đầu vào.
        out_channels: Số hành động.
        device: Thiết bị tính toán.

    Returns:
        Mô hình đã tải trọng số, ở eval mode.
    """
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    if isinstance(checkpoint, dict):
        # state_dict format (checkpoint mới)
        policy_net = create_model(model_type, in_channels, out_channels)
        policy_net.load_state_dict(checkpoint)
        print(f"--> Đã tải state_dict [{model_type.upper()}] từ: {model_path}")
    else:
        # Full model object format (checkpoint cũ, tương thích ngược)
        policy_net = checkpoint
        print(f"--> Đã tải full model object từ: {model_path} (tương thích ngược)")

    return policy_net.to(device).eval()


def play_with_model(
    env: envs.Wrapper,
    policy_net: torch.nn.Module,
    device: torch.device,
    seed: int | None = None,
) -> float:
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    state = torch.tensor(state, device=device, dtype=torch.float32)

    total_reward = 0.0
    while True:
        with torch.no_grad():
            action = policy_net(state.unsqueeze(0)).max(dim=1)[1][0]

        # Ép kiểu action an toàn trước khi truyền vào env
        state, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.tensor(state, device=device, dtype=torch.float32)

        total_reward += float(reward)
        if terminated or truncated:
            break

    return total_reward


def ai_play(model_path: str, model_type: str = "dqn", show_plot: bool = False):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Cần biết in_channels, out_channels: tạo env tạm để lấy thông tin
    env = gym.make("Env-v0", render_mode="human", max_episode_steps=None)
    env = envs.Wrapper(env)

    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n  # type: ignore

    policy_net = load_model(model_path, model_type, in_channels, out_channels, device)

    total_reward = play_with_model(env, policy_net, device)

    print(f"Total reward: {total_reward}, number of frames: {len(env.frames)}")

    env.close()

    if show_plot and len(env.frames) > 0:
        plt.imshow(env.frames[-1])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["human", "ai"])
    parser.add_argument("-m", "--model_path")
    parser.add_argument(
        "--model_type", type=str, default="dqn", choices=["dqn", "dueling"],
        help="Kiến trúc mạng: 'dqn' (mặc định, tương thích model.pth cũ) hoặc 'dueling'"
    )
    parser.add_argument("--plot", action="store_true", help="Show the final frame plot")

    args = parser.parse_args()
    if args.type == "human":
        human_play(show_plot=args.plot)
    else:
        ai_play(args.model_path, model_type=args.model_type, show_plot=args.plot)
