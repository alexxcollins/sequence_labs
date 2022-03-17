import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

# deep enough?
# RestNet consist of blocks and layers
# 4*23 = 92

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


# convolutional layers
# groups=1: all inputs are conv'd with all outputs
def conv5x5(in_planes, out_planes, kernel_size=5, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, kernel_size=3, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False)


# returns a block of 
# norm layer: batch norming
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# same as BB, just with an extra conv+batchnorm+relu step
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #self.expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # this is the whole novelty of ResNet  
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fc_layer=True, zero_init_residual = False):
        self.fc_layer = fc_layer
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # _make_layer functionaslity
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # input size: resized previous layer
        # 2048 * 1 * 1 for ResNet if the image size is 128x128 
        self.out_fc = nn.Linear(2048*1*1, 512)

        if fc_layer:
            self.fc_new = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                torch.nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


	# init residual layers to 0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # x_input: sample_points x batch (num frames) x 3 x H x W
    def forward(self, x_input):
        # must be images 
        try:
           assert x_input.size(2) == 3
        except AssertionError:
           print ("Inputs must be batches of frame sequences")
        # loop over number of frames: 0 to n,  each frame becomes sample_points,3,H,W
        # create a list to stack BATCHES of frames
        batch_of_frames = []
        for frame in range(x_input.size(1)):
            x = self.conv1(x_input[:,frame,:,:,:])
            x = self.bn1(x)
            x = self.relu(x)
            mp = self.maxpool(x)
            l1 = self.layer1(mp)
            l2 = self.layer2(l1)
            l3 = self.layer3(l2)
            out = self.layer4(l3)
            # if output too small: pad it with 0s
            if out.size()[2] < self.avgpool.kernel_size or out.size()[3] < self.avgpool.kernel_size:
               diff_zeros = self.avgpool.kernel_size - out.size()[3]
               if diff_zeros%2:
                  out = F.pad(input=out,pad=(0,diff_zeros,0,diff_zeros))
               else:
                  out = F.pad(input=out,pad=4* (int(diff_zeros/2),))
            out = self.avgpool(out)
            # resize to #batches, #maps * map_h * map_w

            out = out.view(out.size()[0], -1)
            out_fc = self.out_fc(out)
            batch_of_frames.append(out_fc)
        # transpose to get #samples x #frames x #features 
        batch_of_frames=torch.stack(batch_of_frames)
        batch_of_frames.transpose_(0,1)
        return batch_of_frames


# LSTM 
class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=11):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will have batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
     
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        # output: number of classes
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        ####################################################
        # init weights
        for x in self.LSTM.named_parameters():
            if x[0].startswith('weight'):
               torch.nn.init.xavier_uniform_(x[1])
            elif x[0].startswith('bias'):
               x[1].data.fill_(0.01)

        #fully connected layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, x_RNN):
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN)  
        # FC layers, batch first!
        x = self.fc1(RNN_out[:, -1, :])   
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

def rnn_model(**kwargs):
    model = DecoderRNN(CNN_embed_dim = 512, **kwargs)
    return model

