import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAbstractionMSG(nn.Module):
    """Set abstraction layer with multi-scale grouping."""

    def __init__(self, npoint, radii, nsamples, in_channel, mlps):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for mlp in mlps:
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        B, _, N = xyz.shape
        if self.npoint is not None:
            idx = torch.randperm(N)[:self.npoint]
            new_xyz = xyz[:, :, idx].contiguous()
        else:
            new_xyz = xyz
            self.npoint = N

        new_points_list = []
        for radius, nsample, convs, bns in zip(self.radii, self.nsamples,
                                               self.conv_blocks, self.bn_blocks):
            dist = torch.cdist(new_xyz.transpose(1, 2), xyz.transpose(1, 2))
            _, idx = dist.topk(nsample, dim=-1, largest=False)
            grouped_xyz = torch.gather(
                xyz.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                3,
                idx.unsqueeze(1).expand(-1, 3, -1, -1),
            )
            grouped_xyz = (
                grouped_xyz - new_xyz.unsqueeze(-1)
            ).permute(0, 1, 3, 2)

            if points is not None:
                grouped_points = torch.gather(
                    points.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                    3,
                    idx.unsqueeze(1).expand(-1, points.shape[1], -1, -1),
                )
                new_points = torch.cat(
                    [grouped_xyz, grouped_points.permute(0, 1, 3, 2)], dim=1
                )
            else:
                new_points = grouped_xyz

            for conv, bn in zip(convs[:-1], bns[:-1]):
                new_points = F.relu(bn(conv(new_points)))
            new_points = bns[-1](convs[-1](new_points))
            new_points = torch.max(new_points, 2)[0]
            new_points_list.append(new_points)

        new_points = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points


class PointNet2MSGClassifier(nn.Module):
    """PointNet++ classification network with MSG."""

    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg["pointnet2_model"]
        self.input_dim = pointnet2_cfg["input_dim"]
        feature_dim = self.input_dim - 3
        output_num_class = pointnet2_cfg["num_classes"]

        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            in_channel=feature_dim,
            mlps=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            in_channel=320,
            mlps=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        self.sa3 = SetAbstractionMSG(
            npoint=None,
            radii=[None],
            nsamples=[1],
            in_channel=640,
            mlps=[[256, 512, 1024]],
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_num_class)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape}")
        if x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)
        B, C, N = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {C}")

        xyz = x[:, :3, :]
        points = x[:, 3:, :]

        xyz1, points1 = self.sa1(xyz, points)
        xyz2, points2 = self.sa2(xyz1, points1)
        _, points3 = self.sa3(xyz2, points2)

        x = torch.max(points3, dim=2)[0]

        if x.size(0) == 1 and self.training:
            x = torch.cat([x, x], dim=0)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x[:1]
        else:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x
        