import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import crop
from utils.utils import get_preds_fromhm
import numpy as np

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)







class FastFAN(nn.Module):

    def __init__(self, num_modules = 1,depth = 2,imp = 0,device = 0):
        super(FastFAN, self).__init__()
        self.batch_size =1

        self.imp=imp
        self.device = device
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, depth, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            if hg_module == self.num_modules - 1 and imp!=0:
                self.add_module('leasy' + str(hg_module), nn.Conv2d(256, imp, kernel_size=1, stride=1, padding=0))
                self.add_module('lhard' + str(hg_module),
                                nn.Conv2d(256+imp, 68-imp, kernel_size=1, stride=1, padding=0))
            else:
                self.add_module('l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))


            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(self.lastFnum, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))



    def getLandmarksFromFrame(self,image,detected_faces):
        if len(detected_faces) == 0:
            return []
        centers = []
        scales = []
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / 195
            centers.append(center)
            scales.append(scale)
            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()
            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)
            if i == 0:
                imgs = inp
            else:
                imgs = torch.cat((imgs,inp), dim=0)

        out = self.forward(imgs)
        out = out[-1].cpu()
        pts, pts_img = get_preds_fromhm(out, centers, scales)
        #pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
        #landmarks.append(pts_img.numpy())
        return pts_img.numpy().tolist()

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = self._modules['conv_last' + str(i)](ll)
            ll = self._modules['bn_end' + str(i)](ll)
            ll = F.relu(ll, True)


            # Predict heatmaps
            if i == (self.num_modules-1) and self.imp!=0:
                easyp = self._modules['leasy' + str(i)](ll)
                hardp = self._modules['lhard' + str(i)](torch.cat((ll,easyp),1))
                tmp_out = torch.cat((easyp,hardp),1)
                outputs.append(tmp_out)
            else:
                tmp_out = self._modules['l' + str(i)](ll)
                outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll_ = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll_ + tmp_out_

        return outputs

def generateFastFan(modelPath,deviceID):
    torch.cuda.set_device(deviceID)
    model = FastFAN(num_modules=1,depth=2, imp=42,device = deviceID)
    checkpoint = torch.load(
        modelPath,
        map_location=lambda storage, loc: storage.cuda(
            torch.cuda.current_device()
        )
    )
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['state_dict']
    matched_dict = {}
    for k, v in checkpoint_dict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            matched_dict[k] = v

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()

    return model