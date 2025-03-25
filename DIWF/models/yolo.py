import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *

from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=6, nc=None, fusion_num = 4, fusion_layers=(6,11,16,23),
                 onemodaility_layers=(0, 2, 4, 7, 9, 12, 14, 17, 19, 21)):  # model, input channels, number of classes
        self.fusion_num = fusion_num
        self.fusion_layers = fusion_layers
        self.onemodaility_layers = onemodaility_layers
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        # self.fusion_num = fusion_num
        # self.fusion_layers = fusion_layers
        # self.onemodaility_layers = onemodaility_layers

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):

        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):

        y, dt = [], []  # outputs
        m_ = self.model[0]

        for i, m in enumerate(self.model):

            if i < self.fusion_layers[-1] and i not in self.fusion_layers[:self.fusion_num - 1]:
                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
                if i in self.onemodaility_layers:
                    m_ = m
                else:

                    len1 = len(x[0])

                    x_rgb = m(x[:, 0:(len1 // 2)])
                    x_d = m_(x[:, len1 // 2:])

                    x = torch.cat((x_rgb, x_d), 1)

                    y.append(x)
                    y.append(x)  # save output
            elif i in self.fusion_layers[:self.fusion_num - 1]:
                x, merge = m(x)
                y.append(x)
            elif i == self.fusion_layers[-1]:
                x = y[-1]
                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)[1]  # run
                y.append(x)  # save output
            else:
                if m.f != -1 or m.f != -2:  # if not from previous layer
                    # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 or j == -2 else y[j] for j in m.f]  # from earlier layers
                    if isinstance(m.f, int):
                        x = y[m.f]
                    else:
                        t = []
                        for index, j in enumerate(m.f):
                            if j == -1 or j == -2:
                                t.append(x)
                            else:
                                if index != 0:
                                    if j == m.f[index - 1] + 1:
                                        continue
                                t.append(y[j])
                        x = t


                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# class SqueezeAndExcitation(nn.Module):
#     def __init__(self, channel, K, reduction=16):
#         super(SqueezeAndExcitation, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1),
#             Mish(),
#             nn.Conv2d(channel // reduction, K, 1),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         weighting = F.adaptive_avg_pool2d(x, 1)
#         weighting = self.fc(weighting).view(x.shape[0], -1)
#         return self.sigmoid(weighting)
#         # return F.softmax(weighting, -1)

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, K, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.GroupNorm(1, channel // reduction),  # 使用组归一化
            Mish(),
            nn.Conv2d(channel // reduction, K, 1, bias=False),
            nn.GroupNorm(1, K)  # 使用组归一化
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting).view(x.shape[0], -1)
        return self.sigmoid(weighting)

# Adaptive Feature Driven Convolution
class ADFC(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 temprature=30, ratio=16, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight

        self.attention = SqueezeAndExcitation(in_planes, K)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K

        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

# Context Sensitive Dynamic Filtering Module
class CSDF(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16, num_heads=8):
        super(CSDF, self).__init__()
        self.out_planes = out_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            ADFC(in_planes=in_planes, out_planes=out_planes // reduction, kernel_size=1, stride=1, padding=0,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // reduction, out_planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, m):
        b, c, _, _ = m.size()
        y = self.avg_pool(m).view(b, c, 1, 1)
        channel_weight = self.fc(y)
        out = channel_weight * x
        return out

class ImportanceWeightedFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super(ImportanceWeightedFusion, self).__init__()
        self.in_channels = in_channels

        # Define a small network to compute the importance weights
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels * 2, 128),  # Assuming we concatenate rgb and hha features
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # Output two weights for rgb and hha respectively
        )


    def forward(self, rgb, hha):
        # Concatenate rgb and hha along the channel dimension
        x = torch.cat([rgb, hha], dim=1)

        # Use Global Average Pooling to get a vector representation
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        # Get the importance weights
        weights = self.fc(x)

        # Extract the individual weights for rgb and hha
        rgb_weight, hha_weight = weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3), weights[:, 1].unsqueeze(
            1).unsqueeze(2).unsqueeze(3)

        # Weighted sum of rgb and hha
        fused_output = rgb_weight * rgb + hha_weight * hha

        return fused_output

class Fusion(nn.Module):
    def __init__(self, in_planes, reduction=16, bn_momentum=0.0003, num_heads=8):
        super().__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.msfa_rgb = CSDF(in_planes * 2, in_planes)
        self.msfa_hha = CSDF(in_planes * 2, in_planes)

        # 不同尺度的卷积层
        self.conv1x1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_planes, in_planes, kernel_size=5, stride=1, padding=2)

        self.importance_weighted_fusion = ImportanceWeightedFusion(in_channels=in_planes)

        self.conv_gate_rgb = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv_gate_hha = nn.Conv2d(in_planes, in_planes, kernel_size=1)

        self.rgb_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，范围为[0,1]
        self.hha_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，范围为[0,1]

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        len1 = len(x[0])
        rgb = x[:, 0:(len1 // 2)]
        hha = x[:, len1 // 2:]

        # 多尺度特征提取
        rgb_1x1 = self.conv1x1(rgb)
        rgb_3x3 = self.conv3x3(rgb)
        rgb_5x5 = self.conv5x5(rgb)

        # 融合多尺度特征
        rgb = rgb_1x1 + rgb_3x3 + rgb_5x5

        # 对 HHA 执行相同操作
        hha_1x1 = self.conv1x1(hha)
        hha_3x3 = self.conv3x3(hha)
        hha_5x5 = self.conv5x5(hha)
        hha = hha_1x1 + hha_3x3 + hha_5x5

        # Gating
        gate_rgb = torch.sigmoid(self.conv_gate_rgb(rgb))
        gate_hha = torch.sigmoid(self.conv_gate_hha(hha))
        rgb = gate_rgb * rgb
        hha = gate_hha * hha

        rec_rgb = self.msfa_rgb(rgb, x)
        rec_hha = self.msfa_hha(hha, x)

        merge_feature = self.importance_weighted_fusion(rec_rgb, rec_hha)

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)
        return torch.cat([rgb_out, hha_out], dim=1), merge_feature

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!unsure
    ch[-1] //= 2
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        elif m is Fusion:
            c2 = args[0]
            c2 = int(c2 * gw)
            args[0] = c2
        # elif m is ESA:
        #     c2 = args[0]
        #     c2 = int(c2 * gw)
        #     args[0] = c2
        else:
            c2 = ch[f if f < 0 else f + 1]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        layers.append(m_)
        ch.append(c2)
    # print(nn.Sequential(*layers))
    # print(save)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='RGB-D yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # img = torch.rand( 1, 6, 640, 640).to(device)
    # print(img)

    # Create model
    model = Model(opt.cfg, ch=6).to(device)

    # model.train()
    print(model)

    # Profile

    img = torch.rand(2 if torch.cuda.is_available() else 1, 6, 640, 640).to(device)
    y = model(img, profile=False)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
