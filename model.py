import torch
import torch.nn as nn


class DQN(nn.Module):
    """Standard DQN network - giữ nguyên để tương thích ngược với model.pth cũ."""

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DuelingDQN(nn.Module):
    """Dueling DQN network.

    Tách mạng thành 2 luồng:
    - Value stream V(s): ước lượng giá trị của trạng thái
    - Advantage stream A(s, a): ước lượng lợi thế của từng hành động so với trung bình

    Kết hợp: Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))

    Kiến trúc này đặc biệt hiệu quả với game runner vì phần lớn thời gian
    giá trị trạng thái V(s) không phụ thuộc vào hành động (đường chạy trống).

    References:
        Wang et al. (2016) - "Dueling Network Architectures for Deep RL"
        https://arxiv.org/abs/1511.06581
    """

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Shared feature extractor (giữ nguyên kiến trúc conv từ DQN gốc)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Value stream: V(s) - scalar
        self.value_stream = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream: A(s, a) - vector với kích thước = số hành động
        self.advantage_stream = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)

        value = self.value_stream(x)          # shape: (batch, 1)
        advantage = self.advantage_stream(x)  # shape: (batch, n_actions)

        # Kết hợp V và A theo công thức Wang et al. (2016)
        # Trừ đi mean advantage để tránh unidentifiable problem
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q


def create_model(model_type: str, in_channels: int, out_channels: int) -> nn.Module:
    """Factory function để khởi tạo mô hình theo loại chỉ định.

    Args:
        model_type: Loại mô hình - 'dqn' hoặc 'dueling'.
        in_channels: Số kênh đầu vào (bằng k trong frame-stacking).
        out_channels: Số hành động (kích thước action space).

    Returns:
        Mô hình DQN hoặc DuelingDQN.

    Raises:
        ValueError: Nếu model_type không hợp lệ.
    """
    if model_type == "dqn":
        return DQN(in_channels, out_channels)
    elif model_type == "dueling":
        return DuelingDQN(in_channels, out_channels)
    else:
        raise ValueError(
            f"Loại model không hợp lệ: '{model_type}'. Chọn 'dqn' hoặc 'dueling'."
        )
