import argparse
from PIL import Image
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import envs
from play import play_with_model, load_model
from tqdm import tqdm

if __name__ == "__main__":
    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path")
    parser.add_argument("-n", "--n_runs", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=510)
    parser.add_argument("-e", "--episodes", type=int, nargs="+")
    parser.add_argument("--save_play", action="store_true")
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument(
        "--model_type", type=str, default="dqn", choices=["dqn", "dueling"],
        help="Kiến trúc mạng của model cần eval: 'dqn' hoặc 'dueling'"
    )
    args = parser.parse_args()

    env = gym.make("Env-v0", render_mode="rgb_array")
    env = envs.Wrapper(env)

    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n  # type: ignore

    progress_bar = tqdm(total=args.n_runs * len(args.episodes), desc="Progress")

    avg_rewards = []
    avg_frame_counts = []
    # Chạy thực nghiệm trên các model sau khi huấn luyện được N episodes
    for milestone in args.episodes:
        model_path = os.path.join(args.result_path, f"model-{milestone}.pth")
        policy_net = load_model(model_path, args.model_type, in_channels, out_channels, device)

        rewards = []
        frame_counts = []
        for i in range(args.n_runs):
            total_reward = play_with_model(
                env, policy_net, device, seed=(i + 1) * args.seed
            )
            rewards.append(total_reward)
            frame_counts.append(len(env.frames))

            if args.save_play and i == 0:
                # Only save play of the first run
                frames = [Image.fromarray(obs, "RGB") for obs in env.frames]

                frames[0].save(
                    f"experiments/model-{milestone}.gif",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=100,
                    loop=0,
                )

            progress_bar.update(1)

        avg_rewards.append(np.array(rewards, dtype=np.float32).mean())
        avg_frame_counts.append(np.array(frame_counts, dtype=np.float32).mean())

    # Draw plots to see average total rewards and average number of frames
    # each model got over `n_runs` runs.
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(
        args.episodes,
        avg_rewards,
        marker=".",
        markeredgecolor="red",
        markerfacecolor="red",
    )
    axs[0].set_title("Average total reward")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Reward")

    axs[1].plot(
        args.episodes,
        avg_frame_counts,
        marker=".",
        markeredgecolor="red",
        markerfacecolor="red",
    )
    axs[1].set_title("Average number of frames")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Frames")

    plt.subplots_adjust(wspace=0.5)
    if args.save_plot:
        plt.savefig("experiments/plot.png")
    plt.show()
