"""  
地形编码器  
==========  
  
用于高层策略的地形感知  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from typing import Tuple, Optional  
  
  
class TerrainEncoder2D(nn.Module):  
    """  
    2D地形编码器  
      
    处理俯视图地形网格  
    """  
      
    def __init__(self,  
                 input_channels: int = 1,  
                 grid_size: int = 32,  
                 output_dim: int = 128):  
        """  
        初始化  
          
        Args:  
            input_channels: 输入通道数  
            grid_size: 网格大小  
            output_dim: 输出特征维度  
        """  
        super().__init__()  
          
        self.input_channels = input_channels  
        self.grid_size = grid_size  
        self.output_dim = output_dim  
          
        # CNN编码器  
        self.encoder = nn.Sequential(  
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Flatten(),  
        )  
          
        # 计算展平后的维度  
        with torch.no_grad():  
            dummy = torch.zeros(1, input_channels, grid_size, grid_size)  
            flat_dim = self.encoder(dummy).shape[1]  
              
        self.fc = nn.Sequential(  
            nn.Linear(flat_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, output_dim)  
        )  
          
    def forward(self, terrain_grid: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            terrain_grid: 地形网格 [batch, channels, H, W]  
              
        Returns:  
            features: 地形特征 [batch, output_dim]  
        """  
        x = self.encoder(terrain_grid)  
        return self.fc(x)  
  
  
class TerrainEncoder3D(nn.Module):  
    """  
    3D地形编码器  
      
    处理体素化的3D地形  
    """  
      
    def __init__(self,  
                 input_channels: int = 1,  
                 grid_size: int = 16,  
                 output_dim: int = 128):  
        """  
        初始化  
          
        Args:  
            input_channels: 输入通道数  
            grid_size: 网格大小（假设各维度相同）  
            output_dim: 输出特征维度  
        """  
        super().__init__()  
          
        self.input_channels = input_channels  
        self.grid_size = grid_size  
        self.output_dim = output_dim  
          
        # 3D CNN编码器  
        self.encoder = nn.Sequential(  
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Flatten(),  
        )  
          
        # 计算展平后的维度  
        with torch.no_grad():  
            dummy = torch.zeros(1, input_channels, grid_size, grid_size, grid_size)  
            flat_dim = self.encoder(dummy).shape[1]  
              
        self.fc = nn.Sequential(  
            nn.Linear(flat_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, output_dim)  
        )  
          
    def forward(self, terrain_voxel: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            terrain_voxel: 3D体素网格 [batch, channels, D, H, W]  
              
        Returns:  
            features: 地形特征 [batch, output_dim]  
        """  
        x = self.encoder(terrain_voxel)  
        return self.fc(x)  
  
  
class PointCloudEncoder(nn.Module):  
    """  
    点云编码器  
      
    处理声呐等传感器的点云数据  
    基于PointNet简化版  
    """  
      
    def __init__(self,  
                 input_dim: int = 3,  
                 output_dim: int = 128):  
        """  
        初始化  
          
        Args:  
            input_dim: 每个点的维度（通常是3：x, y, z）  
            output_dim: 输出特征维度  
        """  
        super().__init__()  
          
        self.input_dim = input_dim  
        self.output_dim = output_dim  
          
        # 逐点MLP  
        self.point_mlp = nn.Sequential(  
            nn.Linear(input_dim, 64),  
            nn.ReLU(),  
            nn.Linear(64, 128),  
            nn.ReLU(),  
            nn.Linear(128, 256)  
        )  
          
        # 全局特征  
        self.global_mlp = nn.Sequential(  
            nn.Linear(256, 256),  
            nn.ReLU(),  
            nn.Linear(256, output_dim)  
        )  
          
    def forward(self, points: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            points: 点云 [batch, num_points, input_dim]  
              
        Returns:  
            features: 全局特征 [batch, output_dim]  
        """  
        # 逐点特征  
        point_features = self.point_mlp(points)  # [batch, num_points, 256]  
          
        # 全局最大池化  
        global_features, _ = point_features.max(dim=1)  # [batch, 256]  
          
        # 全局特征  
        return self.global_mlp(global_features)  
  
  
class MultiModalTerrainEncoder(nn.Module):  
    """  
    多模态地形编码器  
      
    融合多种地形感知模态  
    """  
      
    def __init__(self,  
                 use_2d_grid: bool = True,  
                 use_3d_voxel: bool = False,  
                 use_point_cloud: bool = False,  
                 grid_size_2d: int = 32,  
                 grid_size_3d: int = 16,  
                 output_dim: int = 128):  
        """  
        初始化  
          
        Args:  
            use_2d_grid: 是否使用2D网格  
            use_3d_voxel: 是否使用3D体素  
            use_point_cloud: 是否使用点云  
            grid_size_2d: 2D网格大小  
            grid_size_3d: 3D网格大小  
            output_dim: 最终输出维度  
        """  
        super().__init__()  
          
        self.use_2d_grid = use_2d_grid  
        self.use_3d_voxel = use_3d_voxel  
        self.use_point_cloud = use_point_cloud  
        self.output_dim = output_dim  
          
        encoders = []  
        total_dim = 0  
          
        if use_2d_grid:  
            self.encoder_2d = TerrainEncoder2D(  
                grid_size=grid_size_2d,  
                output_dim=64  
            )  
            total_dim += 64  
              
        if use_3d_voxel:  
            self.encoder_3d = TerrainEncoder3D(  
                grid_size=grid_size_3d,  
                output_dim=64  
            )  
            total_dim += 64  
              
        if use_point_cloud:  
            self.encoder_pc = PointCloudEncoder(output_dim=64)  
            total_dim += 64  
              
        # 融合层  
        if total_dim > 0:  
            self.fusion = nn.Sequential(  
                nn.Linear(total_dim, 128),  
                nn.ReLU(),  
                nn.Linear(128, output_dim)  
            )  
        else:  
            self.fusion = nn.Identity()  
              
    def forward(self,  
                grid_2d: Optional[torch.Tensor] = None,  
                voxel_3d: Optional[torch.Tensor] = None,  
                point_cloud: Optional[torch.Tensor] = None) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            grid_2d: 2D地形网格  
            voxel_3d: 3D体素  
            point_cloud: 点云  
              
        Returns:  
            features: 融合后的地形特征  
        """  
        features = []  
          
        if self.use_2d_grid and grid_2d is not None:  
            features.append(self.encoder_2d(grid_2d))  
              
        if self.use_3d_voxel and voxel_3d is not None:  
            features.append(self.encoder_3d(voxel_3d))  
              
        if self.use_point_cloud and point_cloud is not None:  
            features.append(self.encoder_pc(point_cloud))  
              
        if features:  
            fused = torch.cat(features, dim=-1)  
            return self.fusion(fused)  
        else:  
            # 返回零向量  
            batch_size = 1  
            if grid_2d is not None:  
                batch_size = grid_2d.shape[0]  
            elif voxel_3d is not None:  
                batch_size = voxel_3d.shape[0]  
            elif point_cloud is not None:  
                batch_size = point_cloud.shape[0]  
                  
            return torch.zeros(batch_size, self.output_dim)  
