import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def fcn(num, num_classes, init_weights=True):
    if num == 8: return FCN_8s(make_layers(), num_classes, init_weights)
    elif num == 16: return FCN_16s(make_layers(), num_classes, init_weights)
    elif num == 32: return FCN_32s(make_layers(), num_classes, init_weights)
    

def make_layers():
    vgg16 = models.vgg16(pretrained=True)
    features = list(vgg16.features.children())
    classifier = list(vgg16.classifier.children())
    
    conv1 = nn.Sequential(*features[:5])
    conv1[0].padding = (100, 100)
    conv2 = nn.Sequential(*features[5:10])
    conv3 = nn.Sequential(*features[10:17])
    conv4 = nn.Sequential(*features[17:24])
    conv5 = nn.Sequential(*features[24:])
    
    conv6 = nn.Conv2d(512, 4096, kernel_size=(7, 7))
    conv7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1))
    
    w_conv6 = classifier[0].state_dict()
    w_conv7 = classifier[3].state_dict()
    
    conv6.load_state_dict({'weight':w_conv6['weight'].view(4096, 512, 7, 7), 'bias':w_conv6['bias']})
    conv7.load_state_dict({'weight':w_conv7['weight'].view(4096, 4096, 1, 1), 'bias':w_conv7['bias']})

    return [conv1, conv2, conv3, conv4, conv5, conv6, conv7]


def init_upsampling(in_channels, out_channels, kernel_size):
    assert kernel_size[0] == kernel_size[1]
    kernel_size = kernel_size[0]
    
    factor = kernel_size//2
    center = factor - 0.5*(1 + kernel_size%2)
    
    r = 1 - torch.abs(torch.arange(0, kernel_size) - center)/factor
    c = r.view(len(r), -1)
    
    matrix = (c*r).repeat(in_channels, out_channels, 1, 1)
    return matrix


class FCN_8s(nn.Module):
    def __init__(self, layers, num_classes, init_weights):
        super(FCN_8s, self).__init__()
        
        self.conv123 = nn.Sequential(layers[0], layers[1], layers[2])
        self.conv4 = layers[3]
        self.conv567 = nn.Sequential(layers[4],
                                     layers[5], nn.ReLU(), nn.Dropout2d(), 
                                     layers[6], nn.ReLU(), nn.Dropout2d())
        
        self.conv3_score = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        self.conv4_score = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        self.conv567_score = nn.Conv2d(4096, num_classes, kernel_size=(1, 1))
        
        self.up_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.up_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.up_pool5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        pool3 = self.conv123(x)
        pool4 = self.conv4(pool3)
        pool5 = self.conv567(pool4)
        
        score_p3 = self.conv3_score(pool3)
        score_p4 = self.conv4_score(pool4)
        score_p5 = self.conv567_score(pool5)
        
        up_2x = self.up_pool5(score_p5)
        if up_2x.shape != score_p4.shape:
            _, _, u2h, u2w = up_2x.shape
            _, _, p4h, p4w = score_p4.shape
            h, w = (p4h-u2h)//2, (p4w-u2w)//2
            up_2x += score_p4[:, :, h:h+u2h, w:w+u2w]
        else:
            up_2x += score_p4
        
        up_4x = self.up_pool4(up_2x)
        if up_4x.shape != score_p3.shape:
            _, _, u4h, u4w = up_4x.shape
            _, _, p3h, p3w = score_p3.shape
            h, w = (p3h-u4h)//2, (p3w-u4w)//2
            up_4x += score_p3[:, :, h:h+u4h, w:w+u4w]
        else:
            up_4x += score_p3

        output = self.up_8x(up_4x)
        if output.shape != original.shape:
            _, _, outh, outw = output.shape
            _, _, orih, oriw = original.shape
            h, w = (outh-orih)//2, (outw-oriw)//2
            output = output[:, :, h:h+orih, w:w+oriw]

        return output

    def _initialize_weights(self):
        targets = [self.conv3_score, self.conv4_score, self.conv567_score, 
                   self.up_8x, self.up_pool4, self.up_pool5]
        
        for layer in targets:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.ConvTranspose2d):
                weight = init_upsampling(layer.in_channels, layer.out_channels, layer.kernel_size)
                layer.weight = nn.Parameter(weight)
                
                
class FCN_16s(nn.Module):
    def __init__(self, layers, num_classes, init_weights):
        super(FCN_16s, self).__init__()
        
        self.conv1234 = nn.Sequential(layers[0], layers[1], layers[2], layers[3])
        self.conv567 = nn.Sequential(layers[4],
                                     layers[5], nn.ReLU(), nn.Dropout2d(), 
                                     layers[6], nn.ReLU(), nn.Dropout2d())
        
        self.conv4_score = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        self.conv567_score = nn.Conv2d(4096, num_classes, kernel_size=(1, 1))
        
        self.up_16x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)
        self.up_pool5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        pool4 = self.conv1234(x)
        pool5 = self.conv567(pool4)
        
        score_p4 = self.conv4_score(pool4)
        score_p5 = self.conv567_score(pool5)
        
        up_2x = self.up_pool5(score_p5)
        if up_2x.shape != score_p4.shape:
            _, _, u2h, u2w = up_2x.shape
            _, _, p4h, p4w = score_p4.shape
            h, w = (p4h-u2h)//2, (p4w-u2w)//2
            up_2x += score_p4[:, :, h:h+u2h, w:w+u2w]
        else:
            up_2x += score_p4

        output = self.up_16x(up_2x)
        if output.shape != original.shape:
            _, _, outh, outw = output.shape
            _, _, orih, oriw = original.shape
            h, w = (outh-orih)//2, (outw-oriw)//2
            output = output[:, :, h:h+orih, w:w+oriw]

        return output

    def _initialize_weights(self):
        targets = [self.conv4_score, self.conv567_score, self.up_16x, self.up_pool5]
        
        for layer in targets:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.ConvTranspose2d):
                weight = init_upsampling(layer.in_channels, layer.out_channels, layer.kernel_size)
                layer.weight = nn.Parameter(weight)
                
                
class FCN_32s(nn.Module):
    def __init__(self, layers, num_classes, init_weights):
        super(FCN_32s, self).__init__()
        
        self.conv = nn.Sequential(layers[0], layers[1], layers[2], layers[3], layers[4],
                                  layers[5], nn.ReLU(), nn.Dropout2d(), 
                                  layers[6], nn.ReLU(), nn.Dropout2d())
        
        self.conv_score = nn.Conv2d(4096, num_classes, kernel_size=(1, 1))
        
        self.up_32x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        pool5 = self.conv(x)
        score_p5 = self.conv_score(pool5)
        
        output = self.up_32x(score_p5)
        if output.shape != original.shape:
            _, _, outh, outw = output.shape
            _, _, orih, oriw = original.shape
            h, w = (outh-orih)//2, (outw-oriw)//2
            output = output[:, :, h:h+orih, w:w+oriw]

        return output

    def _initialize_weights(self):
        targets = [self.conv_score, self.up_32x]
        
        for layer in targets:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.ConvTranspose2d):
                weight = init_upsampling(layer.in_channels, layer.out_channels, layer.kernel_size)
                layer.weight = nn.Parameter(weight)