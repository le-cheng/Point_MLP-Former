import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, farthest0):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(
        device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def act_(a):
    if a.lower() == 'gelu':
        return nn.GELU()
    elif a.lower() == 'None':
        return nn.Identity()
    elif a.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif a.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif a.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif a.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif a.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)
 
class Conv1DBR(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='relu'):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            act_(activation))

    def forward(self, x):
        return self.n(x)

class Conv2DBR(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='relu'):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            act_(activation))

    def forward(self, x):
        return self.n(x)

class Conv1DBRRes(nn.Module):
    def __init__(self, in_channels, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        out_channels = int(in_channels * res_expansion)
        self.act = act_(activation)
        self.n1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act)
        self.n2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(in_channels))

    def forward(self, x):
        return self.act(self.n2(self.n1(x)) + x)
   

class GenerateGraph(nn.Module):
    def __init__(self, fps_points, neighbors,N,B):
        super().__init__()
        self.fps_points = fps_points
        self.neighbors = neighbors
        self.farthest = torch.randint(0, N, (B,))
        self.register_buffer('farthest', self.farthest)

    def forward(self, xyz, x):
        farthest = self.farthest.to(xyz.device)
        fps_idx = farthest_point_sample(xyz.contiguous(), self.fps_points, farthest)
        
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(x, fps_idx)  # [B, npoint, d]
        
        # knn
        distance = square_distance(new_xyz, xyz)
        idx = distance.topk(self.neighbors, dim=-1, largest=False, sorted=False)[1]
        # idx = distance.topk(self.kneighbors, dim=-1, largest=False)[1]

        grouped_points = index_points(x, idx)  # [B, npoint, k, d]
        edge = grouped_points-new_points.unsqueeze(dim=-2)
        
        return new_xyz, edge

class Channel(nn.Module):
    def __init__(self, in_channels, blocks=1, res_expansion=1, bias=True, activation='relu'):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(Conv1DBRRes(
                in_channels, res_expansion=res_expansion, bias=bias, activation=activation))
        self.resmlp = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.resmlp(x)

class Local(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=1, 
                 res_expansion=1., bias=True, activation='relu'):
        super().__init__()
        self.proj = Conv1DBR(in_channels, out_channels, bias=bias, activation=activation)
        self.channel = Channel(out_channels, blocks, res_expansion, bias, activation)

    def forward(self, x):
        b, n, k, d = x.shape
        x = x.permute(0, 1, 3, 2).view(-1, d, k)
        x = self.channel(self.proj(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = x.view(b, n, -1).permute(0, 2, 1)
        return x


class SimMLP(nn.Module):
    def __init__(self, in_channels, out_channels, N,B, fps_points, kneighbor=20,
                 la_num=1, ca_num=1, res_expansion=1.,  bias=False, act="relu"):
        super().__init__()
        self.gg = GenerateGraph(fps_points, kneighbor,N,B)
        self.la = Local(
            in_channels, out_channels, la_num, res_expansion, bias, act)
        self.ca = Channel(
            out_channels, ca_num, res_expansion, bias, act)

    def forward(self, xyz, x):
        xyz, x = self.gg(xyz, x.permute(0, 2, 1))
        x = self.la(x)
        x = self.ca(x)
        return xyz, x


class Local2Former(nn.Module):
    def __init__(self, x_channels, t_channels, hid_channels, dropout=0.):
        super().__init__()
        self.linear_q = nn.Linear(t_channels, hid_channels, bias=False)
        self.linear_k = nn.Linear(x_channels, hid_channels, bias=False)
        self.linear_v = nn.Linear(x_channels, hid_channels, bias=False)
        self.scale = hid_channels ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(
            nn.Linear(hid_channels, t_channels, bias=False),
            nn.Dropout(dropout))

    def forward(self, x, z):
        q = self.linear_q(z)  # b,m,hid
        k = self.linear_k(x).transpose(1, 2)  # b,hid,n
        v = self.linear_v(x)  # b,n,hid

        a = q @ k * self.scale  # b,m,num
        a = self.softmax(a)
        out = a @ v  # b,m,hid
        return z + self.proj(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_ch, hid_ch, num_heads, att_drop=0., dropout=0., bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_size = hid_ch // num_heads
        self.scale = self.qkv_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        self.linear_qkv = nn.Linear(in_ch, 3*hid_ch, bias=bias)
        self.att_dropout = nn.Dropout(att_drop)
        self.proj = nn.Sequential(
            nn.Linear(hid_ch, in_ch, bias=False),
            nn.Dropout(dropout))

    def forward(self, x, attn_bias=None, mask=None):
        b, num, dim = x.size()
        qkv = self.linear_qkv(x) \
            .view(b, num, 3, self.num_heads, self.qkv_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1].transpose(3, 2), qkv[2]

        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        attn = q.matmul(k) * self.scale
        if mask is not None:
            attn.masked_fill_(mask == 1, -1000)
        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(-3)
        a = self.softmax(attn)
        a = self.att_dropout(a)
        x = a.matmul(v) 

        x = x.transpose(1, 2).contiguous().view(b, num, -1)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_features=None, acti='gelu', bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_ch
        hidden_ch = hidden_ch or in_ch
        self.fc1 = nn.Sequential(
            nn.Linear(in_ch, hidden_ch, bias), act_(acti), nn.Dropout(drop))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_ch, out_features, bias), nn.Dropout(drop))

    def forward(self, x):
        return self.fc2(self.fc1(x))


class EncoderLayer(nn.Module):
    def __init__(self, in_ch, num_heads, mlp_ratio=2., act='gelu',
                 att_drop=0., drop=0., drop_path=0., bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.att = MultiHeadAttention(
            in_ch=in_ch, hid_ch=in_ch, num_heads=num_heads, att_drop=att_drop, dropout=drop)

        self.norm2 = nn.LayerNorm(in_ch)
        self.mlp = Mlp(
            in_ch=in_ch, hidden_ch=int(in_ch * mlp_ratio),
            acti=act, bias=bias, drop=drop)

        self.drop_path = \
            DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.att(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Former(nn.Module):
    def __init__(self, in_ch, num_tf=1, num_heads=2, mlp_ratio=2., act='relu',
                 att_drop=0., drop=0., drop_path=0., bias=False):
        super(Former, self).__init__()
        self.tf = nn.ModuleList()
        for i_layer in range(num_tf):
            layer = EncoderLayer(
                in_ch, num_heads, mlp_ratio=mlp_ratio, act=act,
                drop_path=drop_path[i_layer] if
                isinstance(drop_path, list) else drop_path,
                att_drop=att_drop, drop=drop, bias=bias)
            self.tf.append(layer)

    def forward(self, t):
        for m in self.tf:
            t = m(t)
        return t


class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, N,B, fps_points, k_neighbors,
                 la_num, ca_num, res_expansion=1.0, act='gelu', bias=False,
                 t_channels=256, heads=2, num_tf=2):
        super().__init__()
        self.simmlp = SimMLP(
            in_channels=in_channels, 
            out_channels=out_channels,
            N=N,B=B,
            fps_points=fps_points,
            kneighbor=k_neighbors,
            la_num=la_num,
            ca_num=ca_num,
            res_expansion=res_expansion, 
            bias=bias,
            act=act)

        self.local2former = Local2Former(
            x_channels=in_channels, t_channels=t_channels, hid_channels=t_channels)

        self.former = Former(
            in_ch=t_channels, num_tf=num_tf, num_heads=heads, mlp_ratio=2.,
            act=act, att_drop=0., drop=0.2, drop_path=0.1, bias=bias)

    def forward(self, xyz, x, t):
        t_hid = self.local2former(x.transpose(1, 2), t)
        t = self.former(t_hid)  # b,m,d
        xyz, x = self.simmlp(xyz, x)
        return xyz, x, t


def KNN(x, k):
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=-1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class Input_Embed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=32, k=20, bias=False, act='hardswish'):
        super().__init__()
        self.k = k
        self.conv = Conv2DBR(in_ch*2, embed_dim, bias=bias, activation=act)
        
    def forward(self, xyz):
        x = self.get_xyzgraph(xyz=xyz, k=self.k)
        x = self.conv(x.permute(0, 3, 1, 2))
        x = x.max(-1)[0]
        return xyz, x
    
    def get_xyzgraph(self, xyz, k=20):
        idx = KNN(xyz, k=k)
        grouped_xyz = index_points(xyz, idx)

        xyz_repeat = xyz.unsqueeze(-2).expand(-1, -1, k, -1)
        point_cat = torch.cat((xyz_repeat, grouped_xyz-xyz_repeat), dim=-1)
        return point_cat


class Moduletemp(nn.Module):
    def __init__(self, cfg=None, token=6, t_channels=256, embed_dim=64, fps_points=1024,
                 la_num=[2, 2, 2, 2], ca_num=[2, 2, 2, 2], dim_expansion=[2, 2, 2, 2],
                 reducers=[2, 2, 2, 2], k_neighbors=[32, 32, 32, 32], 
                 activation="gelu", heads=2, num_tf=2, res_expansion=1.0, B=16
                 ):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, token, t_channels))
        trunc_normal_(self.token, std=.02)

        self.input_embed = Input_Embed(
            in_ch=3, embed_dim=embed_dim, k=k_neighbors[0], act=activation)

        self.base_block_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(len(la_num)):
            # channel
            out_channel = last_channel * dim_expansion[i]
            # points
            fps_points = fps_points // reducers[i]
            # mlp_model
            block = BaseBlock(
                in_channels=last_channel,
                out_channels=out_channel,
                N=fps_points*reducers[i],B=B,
                fps_points=fps_points,
                k_neighbors=k_neighbors[i],
                la_num=la_num[i],
                ca_num=ca_num[i],
                res_expansion=res_expansion,
                t_channels=t_channels, heads=heads,
                num_tf=num_tf)

            last_channel = out_channel
            self.base_block_list.append(block)

        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(last_channel*2+t_channels, 512),
            nn.BatchNorm1d(512),
            act_('relu'),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            act_('relu'),
            nn.Dropout(0.5),
            nn.Linear(256, cfg.num_classes)
        )

    def forward(self, xyz):
        b, _, dim = xyz.shape
        assert dim == 3

        t = self.token.expand(b, -1, -1)  # b,m,d
        xyz, x = self.input_embed(xyz)  # (b,npoint,3)(b,embed_dim,npoint)

        for m in self.base_block_list:
            xyz, x, t = m(xyz, x, t)  # b,n,3 | b,c,n | b,m,d

        x_avg = self.avg(x)
        x_max = self.max(x)
        x = torch.cat((x_max, x_avg), dim=1).squeeze()  # b,c
        t = t.max(1)[0]
        out = torch.cat((x, t), -1)

        return self.classifier(out)


def Module(cfg=None):
    return Moduletemp(cfg=cfg, token=16, t_channels=256, 
                      embed_dim=64, fps_points=1024,
                      la_num=[1, 1, 2, 1], 
                      ca_num=[1, 1, 2, 1],
                      dim_expansion=[2, 2, 2, 2],
                      reducers=[2, 2, 2, 2], 
                      k_neighbors=[24, 24, 24, 24], 
                      activation="gelu", 
                      heads=8, 
                      num_tf=1, 
                      res_expansion=1.0,B=cfg.batch_size)


if __name__ == '__main__':
    import sys
    sys.path.append("../")
    from utils import read_yaml
    cfg = read_yaml('../configs/config.yaml')
    data = torch.rand(32, 1024, 3)
    print("===> testing pointMLP ...")
    model = Module(cfg).cuda()
    out = model(data.cuda())
    print(out.shape)
