from numpy import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from alfred.dl.torch.common import print_shape

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'None':
        return nn.Identity()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


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


def farthest_point_sample(xyz, npoint):
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
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)*0
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


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(
        sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=False, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()

        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        # if self.normalize is not None:
        #     add_channel = 3 if self.use_xyz else 0
        #     self.affine_alpha = nn.Parameter(
        #         torch.ones([1, 1, 1, channel + add_channel]))
        #     self.affine_beta = nn.Parameter(
        #         torch.zeros([1, 1, 1, channel + add_channel]))
        # self.pos_embed = nn.Linear(9, channel)

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat(
                [grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-
                                 1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            a = grouped_points-mean
            # std = torch.std(a.reshape(B, -1), dim=-1,
            #                 keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            # grouped_points = a/(std + 1e-5)
            # # grouped_points = grouped_points-mean
            # grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        # position
        # pos = torch.cat((new_xyz.unsqueeze(dim=-2).expand(-1, -1, self.kneighbors, -1),
        #                 grouped_xyz, grouped_xyz-new_xyz.unsqueeze(dim=-2)), dim=-1)
        # pos = self.pos_embed(pos)

        # new_points = torch.cat([a, pos], dim=-1)
        # new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1),pos], dim=-1)
        return new_xyz, a


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = channels
        self.transfer = ConvBNReLU1D(
            in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class Local(nn.Module):
    def __init__(self, in_channels, hid, out_channels, anchor_points, kneighbor=20, pre_block_num=2,
                 pos_block_num=2, groups=1, res_expansion=1., bias=False, activation="relu", use_xyz=False):
        super().__init__()

        self.local_grouper = LocalGrouper(
            in_channels, anchor_points, kneighbor, use_xyz=use_xyz)  # [b,g,k,d]

        self.pre_blocks = PreExtraction(
            in_channels, out_channels, pre_block_num, groups=groups,
            res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)

        self.pos_blocks = PosExtraction(
            out_channels, pos_block_num, groups=groups,
            res_expansion=res_expansion, bias=bias, activation=activation)

        # self.transformer = nn.ModuleList()
        # for i_layer in range(pos_block_num):
        #     m = EncoderLayer(out_channels, 4, mlp_ratio=1, act=activation,
        #                      att_drop=0., drop=0.,
        #                      drop_path=0.1,
        #                      bias=bias)
        #     self.transformer.append(m)

    def forward(self, xyz, x):
        xyz, x = self.local_grouper(
            xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
        x = self.pre_blocks(x)  # [b,d,g]
        x = self.pos_blocks(x)  # [b,d,g]
        # for layer in self.transformer:
        #     x = layer(x.transpose(2, 1), attn_bias=None).transpose(
        #         2, 1)  # (b*ngroup, pergroup_point, dim)
        return xyz, x


class Local2Former(nn.Module):
    """Local2Former
        input:  x (b num c)
                z (b m z_dim)
        output: x (b m z_dim)
    """

    def __init__(self, x_channels, t_channels, hid_channels, dropout=0.):
        super().__init__()
        self.x_channels = x_channels
        self.t_channels = t_channels

        self.linear_q = nn.Linear(t_channels, hid_channels, bias=False)
        self.linear_k = nn.Linear(x_channels, hid_channels, bias=False)
        self.linear_v = nn.Linear(x_channels, hid_channels, bias=False)
        self.scale = hid_channels ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(
            nn.Linear(hid_channels, t_channels, bias=False),
            nn.Dropout(dropout))

    def forward(self, x, z):
        _, _, dim = x.shape
        _, _, z_dim = z.shape
        assert dim == self.x_channels
        assert z_dim == self.t_channels

        q = self.linear_q(z)  # b,m,hid
        k = self.linear_k(x).transpose(1, 2)  # b,hid,n
        v = self.linear_v(x)  # b,n,hid

        a = q @ k * self.scale  # b,m,num
        a = self.softmax(a)
        out = a @ v  # b,m,hid
        return z + self.proj(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_ch, hid_ch, num_heads, att_drop=0., dropout=0., bias=False):
        super(MultiHeadAttention, self).__init__()
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
        q, k, v = qkv[0], qkv[1].transpose(
            3, 2), qkv[2]  # [b, h ,num, qkv_size]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        attn = q.matmul(k) * self.scale  # [b, h, q_len, k_len]

        if mask is not None:  # 屏蔽不想要的输出
            attn.masked_fill_(mask == 1, -inf)  # float(’-inf’)
        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(-3)

        a = self.softmax(attn)
        a = self.att_dropout(a)
        x = a.matmul(v)  # [b, h, q_len, c]

        x = x.transpose(1, 2).contiguous().view(
            b, num, -1)  # [b, q_len, h, attn]
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_features=None, act='gelu', bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_ch
        hidden_ch = hidden_ch or in_ch
        self.act = get_activation(act)
        self.fc1 = nn.Sequential(
            nn.Linear(in_ch, hidden_ch, bias), self.act, nn.Dropout(drop))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_ch, out_features, bias), nn.Dropout(drop))

    def forward(self, x):
        return self.fc2(self.fc1(x))


class EncoderLayer(nn.Module):
    """MultiHeadAttention+FFN

        inputs: b n C
        output: b n C
    """

    def __init__(self, in_ch, num_heads, mlp_ratio=2., act='gelu',
                 att_drop=0., drop=0., drop_path=0., bias=False):
        super(EncoderLayer, self).__init__()
        self.in_ch = in_ch

        self.norm1 = nn.LayerNorm(in_ch)
        self.att = MultiHeadAttention(
            in_ch=in_ch, hid_ch=in_ch, num_heads=num_heads, att_drop=att_drop, dropout=drop)

        self.norm2 = nn.LayerNorm(in_ch)
        self.mlp = Mlp(
            in_ch=in_ch, hidden_ch=int(in_ch * mlp_ratio),
            act=act, bias=bias, drop=drop)

        self.drop_path = \
            DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_bias=None):
        _, _, dim = x.shape
        assert dim == self.in_ch
        x = x + self.drop_path(self.att(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Former(nn.Module):
    """MHFormer

    inputs: b m z_dim
    output: b m z_dim
    """

    def __init__(self, in_ch, num_layers=1, num_heads=2, mlp_ratio=2., act='relu',
                 att_drop=0., drop=0., drop_path=0., bias=False):
        super(Former, self).__init__()
        self.tf = nn.ModuleList()
        for i_layer in range(num_layers):
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
    def __init__(self, in_channels, out_channels, anchor_points, k_neighbors,
                 pre_block_num, pos_block_num, res_expansion=1.0, activation='gelu', groups=1, bias=False,
                 use_xyz=False, t_channels=128, heads=2, num_formerlayers=2):
        super(BaseBlock, self).__init__()

        self.local = Local(
            in_channels=in_channels, hid=None, out_channels=out_channels,
            anchor_points=anchor_points,
            kneighbor=k_neighbors,
            pre_block_num=pre_block_num,
            pos_block_num=pos_block_num,
            groups=groups, res_expansion=res_expansion, bias=bias,
            activation=activation, use_xyz=use_xyz)

        self.local2former = Local2Former(
            x_channels=in_channels, t_channels=t_channels, hid_channels=t_channels, dropout=0.)

        self.former = Former(
            in_ch=t_channels, num_layers=num_formerlayers, num_heads=heads, mlp_ratio=2.,
            act='relu', att_drop=0., drop=0.2, drop_path=0.1, bias=False)

    def forward(self, xyz, x, t):
        # print_shape(x)
        t_hid = self.local2former(x.transpose(1, 2), t)  # b,m,d
        t = self.former(t_hid)  # b,m,d
        xyz, x = self.local(xyz, x)
        
        return xyz, x, t


def get_graph_feature(xyz, k=20, npoint=1024, idx=None):
    if xyz.shape[-2] == npoint:
        new_xyz = xyz
        if idx is None:
            idx = knn_T(xyz, k=k)
        grouped_xyz = index_points(xyz, idx)
    else:
        new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
        # idx = query_ball_point(radius, k, xyz, new_xyz)
        dists = square_distance(new_xyz, xyz)  # B x nsample x num
        idx = dists.argsort()[:, :, :k]

        grouped_xyz = index_points(xyz, idx)
    xyz_repeat = new_xyz.unsqueeze(-2).expand(-1, -1, k, -1)
    # point_cat = torch.cat((xyz_repeat, grouped_xyz, grouped_xyz-xyz_repeat), dim=-1)
    point_cat = torch.cat((xyz_repeat, grouped_xyz-xyz_repeat), dim=-1)
    return point_cat, new_xyz


def knn_T(x, k):  # [b,n,k]
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=-1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class Conv2dBR(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False, activation='leakyrelu'):
        super().__init__()
        self.act = get_activation(activation)
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_ch),
            self.act)

    def forward(self, x):
        return self.layers(x)


class Conv1dBR(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, activation='leakyrelu'):
        super(Conv1dBR, self).__init__()
        self.act = get_activation(activation)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act)

    def forward(self, x):
        return self.layers(x)


class Input_Embed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=32, npoint=512, k=20, bias=False, activation='hardswish', max_=True, agg=False):
        super().__init__()
        self.k = k
        self.max_ = max_
        self.npoint = npoint
        self.agg = agg
        if self.agg:
            self.conv = Conv2dBR(in_ch*2, embed_dim,
                                 bias=bias, activation=activation)
        else:
            self.conv = Conv1dBR(
                in_ch, embed_dim, bias=bias, activation=activation)

    def forward(self, xyz, idx=None):
        b, num, dim = xyz.shape
        if dim is not 3:
            xyz = xyz.transpose(2, 1)
            b, num, dim = xyz.shape

        if self.agg:
            x, xyz = get_graph_feature(
                xyz=xyz, k=self.k, npoint=self.npoint, idx=idx)  # (b,num,k,3*3)
            x = self.conv(x.permute(0, 3, 1, 2))
            if self.max_:
                x = x.max(-1)[0]  # (b,embed_dim,npoint)
            else:
                x = x.mean(dim=-1, keepdim=False)
        else:
            x = self.conv(xyz.transpose(2, 1))
        return xyz, x


class Moduletemp(nn.Module):
    def __init__(self, cfg=None, token=6, t_channels=256, embed_dim=64, npoint=1024,
                 pre_blocks=[2, 2, 2, 2], dim_expansion=[2, 2, 2, 2],
                 reducers=[2, 2, 2, 2], k_neighbors=[32, 32, 32, 32], pos_blocks=[2, 2, 2, 2],
                 activation="relu", heads=2, num_formerlayers=2, res_expansion=1.0
                 ):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, token, t_channels))
        trunc_normal_(self.token, std=.02)

        self.input_embed = Input_Embed(
            in_ch=3, embed_dim=embed_dim, npoint=npoint, k=k_neighbors[0],
            activation=activation, max_=True, agg=True)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        self.base_block_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = npoint
        for i in range(len(pre_blocks)):
            # channel
            out_channel = last_channel * dim_expansion[i]
            # points
            anchor_points = anchor_points // reducers[i]
            # mlp_model
            block = BaseBlock(
                in_channels=last_channel,
                out_channels=out_channel,
                anchor_points=anchor_points,
                k_neighbors=k_neighbors[i],
                pre_block_num=pre_blocks[i],
                pos_block_num=pos_blocks[i],
                res_expansion=1.0,
                t_channels=t_channels, heads=heads,
                num_formerlayers=num_formerlayers)

            last_channel = out_channel
            self.base_block_list.append(block)

        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(last_channel*2+t_channels, 512),
            nn.BatchNorm1d(512),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(256, 15)
        )

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1).contiguous()
        b, dim, num = xyz.shape
        assert dim == 3

        t = self.token.repeat(b, 1, 1)  # b,m,d
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
                      embed_dim=64, npoint=1024,
                      pre_blocks=[1, 1, 2, 1], 
                      dim_expansion=[2, 2, 2, 2],
                      reducers=[2, 2, 2, 2], 
                      k_neighbors=[24, 24, 24, 24], 
                      pos_blocks=[1, 1, 2, 1],
                      activation="relu", 
                      heads=8, 
                      num_formerlayers=1, 
                      res_expansion=1.0)


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
