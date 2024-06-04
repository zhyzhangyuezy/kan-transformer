import torch
import torch.nn as nn
from fftKAN import *
from effKAN import *

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量

        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        return out


class LSTM_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_ekan, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量

        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.e_kan = KAN([hidden_dim, output_dim])

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        return out

class GRU_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.e_kan = KAN([hidden_dim,10, output_dim])
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class BiLSTM_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.e_kan = KAN([hidden_dim* 2, 10, output_dim])

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.e_kan(out[:, -1, :])
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2,use_kan=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_outputs)  # 修改以匹配输出特征数

    def forward(self, x):
        x = x.transpose(1, 2)  # 将 batch_size, sequence_length, num_features 转换为 batch_size, num_features, sequence_length
        x = self.network(x)
        x = x[:, :, -1]  # 选择每个序列的最后一个输出
        x = self.fc(x)
        return x

class TemporalConvNet_ekan(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2,use_kan=True):
        super(TemporalConvNet_ekan, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.e_kan = KAN([num_channels[-1],10, num_outputs])

    def forward(self, x):
        x = x.transpose(1, 2)  # 将 batch_size, sequence_length, num_features 转换为 batch_size, num_features, sequence_length
        x = self.network(x)
        x = x[:, :, -1]  # 选择每个序列的最后一个输出
        x = self.e_kan(x)
        return x
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        ##Chomp1d 是一个简单的自定义层，用于剪切掉因为填充(padding)导致的多余的输出，这是保证因果卷积不看到未来信息的关键。
        return x[:, :, :-self.padding]





class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs,hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space=hidden_space

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.output_layer = nn.Linear(hidden_space, num_outputs)
        self.transform_layer=nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]

        # 全连接层生成最终输出
        x = self.output_layer(x)
        return x

class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs,hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer_ekan, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space=hidden_space

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.e_kan = KAN([hidden_space, 10, num_outputs])
        self.transform_layer=nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]

        # 全连接层生成最终输出
        x = self.e_kan(x)
        return x



