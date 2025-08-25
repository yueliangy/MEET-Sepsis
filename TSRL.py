import torch
import copy
from torch import nn
import numpy as np

device = 'cuda'

class Config:
    seed = 42
    projection_dim = 35
    batch_size = 64
    lr = 0.005
    pretrain_epochs = 100
    num_views = 35  # 视图数量
    view_dim = 3  # 每个视图特征维度

    # CNN参数
    cnn_params = {
        'filters': [64, 128],
        'kernel_sizes': [5, 3],
        'pool_size': 2,
        'dropout_rate': 0.4
    }


class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output

class TSRL(nn.Module):
    def __init__(self, input_dim, view_dim, num_views):
        super().__init__()
        self.num_views = num_views
        self.view_dim = view_dim

        # 视图划分模块
        self.view_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_views * view_dim,
                kernel_size=3,
                padding=1,
                groups=input_dim
            ),
            nn.BatchNorm1d(num_views * view_dim),
            nn.GELU()
        )

        self.view_weights = nn.Parameter(torch.ones(num_views, requires_grad=True))

        # CNN处理
        self.cnn_blocks = nn.ModuleList()
        for _ in range(num_views):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=view_dim,
                    out_channels=Config.cnn_params['filters'][0],
                    kernel_size=Config.cnn_params['kernel_sizes'][0],
                    padding=Config.cnn_params['kernel_sizes'][0] // 2
                ),
                nn.BatchNorm1d(Config.cnn_params['filters'][0]),
                nn.GELU(),
                nn.MaxPool1d(Config.cnn_params['pool_size']),
                nn.Dropout(Config.cnn_params['dropout_rate']),

                nn.Conv1d(
                    in_channels=Config.cnn_params['filters'][0],
                    out_channels=Config.cnn_params['filters'][1],
                    kernel_size=Config.cnn_params['kernel_sizes'][1],
                    padding=Config.cnn_params['kernel_sizes'][1] // 2
                ),
                nn.BatchNorm1d(Config.cnn_params['filters'][1]),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.cnn_blocks.append(block)

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.cnn_params['filters'][1],
            num_heads=8,
            batch_first=True
        )
        # 最终投影层
        self.final_proj = nn.Linear(
            Config.cnn_params['filters'][1]*num_views,
            Config.projection_dim
        )

    def forward(self, x):
        # 输入维度验证
        assert x.dim() == 3, f"输入需要是3D张量，得到{x.dim()}D"
        # 视图划分
        batch_size, seq_len, input_dim = x.shape
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.view_conv(x)  # [batch, num_views*view_dim, seq_len]

        # 调整维度
        x = x.view(batch_size, self.num_views, self.view_dim, seq_len)
        x = x.permute(0, 1, 3, 2)  # [batch, views, seq_len, view_dim]

        view_features = []
        for i in range(self.num_views):
            view = x[:, i, :, :]  # [batch, seq_len, view_dim]
            # CNN处理
            view = view.permute(0, 2, 1)  # [batch, view_dim, seq_len]
            cnn_out = self.cnn_blocks[i](view)  # [batch, 128, 1]
            # 调整维度
            cnn_out = cnn_out.permute(0, 2, 1)  # [batch, 1, 128]
            # 自注意力
            attn_out, _ = self.attention(cnn_out, cnn_out, cnn_out)
            view_features.append(attn_out.squeeze(1))  # [batch, 128]

        fused = torch.cat(view_features, dim=-1)
        features = self.final_proj(fused)
        return features

def pretrain_model(model, train_data, device):
    model.train()
    train_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.MSELoss()

    print("\n=== 开始预训练 ===")
    for epoch in range(Config.pretrain_epochs):
        total_loss = 0
        indices = np.random.permutation(train_data.shape[0])
        for i in range(0, len(indices), Config.batch_size):
            batch_indices = indices[i:i + Config.batch_size]
            batch = train_data[batch_indices]

            # 输入数据
            inputs = torch.tensor(batch, dtype=torch.float32).to(device)

            # 模型预测
            optimizer.zero_grad()
            predictions = model(inputs)

            loss = criterion(predictions.unsqueeze(1).expand(-1, inputs.shape[1], -1), inputs)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(indices) / Config.batch_size)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{Config.pretrain_epochs}], Loss: {avg_loss:.4f}")
    return model

