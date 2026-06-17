import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LaneCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 128x128 RGB camera input + lidar + state.

    Architecture (camera branch):
        Conv(3 → 32,  k=3, s=1) → BN → ReLU   # fine edges, color blobs; no stride loss
        Conv(32 → 64, k=3, s=2) → BN → ReLU   # downsample x2: 128 → 64
        Conv(64 → 128,k=3, s=2) → BN → ReLU   # downsample x2:  64 → 32
        Conv(128→ 128,k=3, s=2) → BN → ReLU   # downsample x2:  32 → 16
        Conv(128→ 256,k=3, s=2) → BN → ReLU   # downsample x2:  16 →  8
        AdaptiveAvgPool(4x4) → Flatten → Linear(256*16, 512) → ReLU
    """

    def __init__(self, observation_space, features_dim: int = 512):
        # features_dim here is only for the camera branch output.
        # CombinedExtractor will concatenate lidar and state on top.
        super().__init__(observation_space, features_dim=features_dim)

        cam_space = observation_space.spaces["camera"]
        n_input_channels = cam_space.shape[2]  # 3 (RGB) or 1 (grayscale)

        self.cnn = nn.Sequential(
            # Layer 1: stride-1 preserves full 128x128 spatial map
            nn.Conv2d(n_input_channels, 32,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 2: 128 → 64
            nn.Conv2d(32,  64,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 3: 64 → 32
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 4: 32 → 16
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 5: 16 → 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Pool to fixed 4x4 so the fc size is independent of input resolution
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Compute actual flattened size after conv stack
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, cam_space.shape[0], cam_space.shape[1])
            cnn_out_dim = self.cnn(sample).shape[1]  # 256 * 4 * 4 = 4096

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

        # Lidar branch: small MLP
        lidar_dim = observation_space.spaces["lidar"].shape[0]
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # State branch (speed scalar)
        state_dim = observation_space.spaces["state"].shape[0]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
        )

        # Total output dim fed to the policy/value heads
        self._features_dim = features_dim + 64 + 16

    def forward(self, observations: dict) -> torch.Tensor:
        # Camera: SB3 passes uint8 images — normalise to [0,1] here
        img = observations["camera"].float() / 255.0
        # SB3 Dict spaces give (B, H, W, C); Conv2d needs (B, C, H, W)
        img = img.permute(0, 3, 1, 2)

        cam_features   = self.linear(self.cnn(img))
        lidar_features = self.lidar_net(observations["lidar"].float())
        state_features = self.state_net(observations["state"].float())

        return torch.cat([cam_features, lidar_features, state_features], dim=1)
