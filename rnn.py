
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # <-- Make sure to import this

# -----------------------------------------------------------------
# This is the new ResNet-based Bottom Model
# -----------------------------------------------------------------
class BottomModel_ResNet(nn.Module):
    def __init__(self, gpu=False):
        super(BottomModel_ResNet, self).__init__()
        self.gpu = gpu

        # Load a ResNet-18 model
        # We use 'weights=None' because we'll be training from scratch
        resnet = models.resnet18(weights=None) 

        # We'll use the "stem" (conv1, bn1, relu, maxpool) and layer1.
        # - resnet.conv1 + resnet.maxpool performs two 2x downsamples (total 4x)
        # - The output of layer1 in ResNet-18 has 64 channels.
        # This perfectly matches your original BottomModel's output:
        # (H, W) -> (H/4, W/4) and 64 output channels.
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1 # layer1 consists of two residual blocks
        )
        
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda() 
        
        # Pass the input chunk through our ResNet layers
        x = self.features(x)
        return x

# -----------------------------------------------------------------
# This TopModel is UNCHANGED from your original code
# -----------------------------------------------------------------
class TopModel(nn.Module):
    def __init__(self,gpu=False,input_size=8):
        super(TopModel,self).__init__()
        self.gpu = gpu
        # The linear layer nn.Linear(64*input_size*8, 256)
        # expects 64 channels, an 'input_size' (which is the
        # concatenated height), and a width of 8.
        # Our new BottomModel_ResNet provides this exact shape.
        self.linear = nn.Linear(64*input_size*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        if gpu:
            self.cuda()

    def forward(self,x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.linear(x.view(B,-1)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

# -----------------------------------------------------------------
# This Model class is UPDATED to use the new BottomModel_ResNet
# -----------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, gpu=False,multies=2,unit = 0.25):
        super(Model, self).__init__()
        self.gpu = gpu
        self.multies = multies
        self.unit = unit
        self.other_unit = (1-unit)/(multies-1)
        
        # *** CHANGED ***
        # We now create a list of our new ResNet-based bottom models
        self.models = nn.ModuleList([BottomModel_ResNet(gpu) for i in range(self.multies)])
        
        # *** SIMPLIFIED ***
        # The original complex calculation for input_size was likely
        # from an old model version without padding.
        # Both your padded CNN and this ResNet model perform
        # two 2x downsamplings, resulting in H/4.
        # Assuming a 32x32 input image (common for 10 classes):
        input_height = 32
        output_height = input_height // 4 # (32 -> 16 -> 8)
        
        # We pass this simplified, correct height (8) to the TopModel
        self.top = TopModel(gpu, input_size=output_height)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        
        # This splitting logic is UNCHANGED
        x_list = x.split([int(x.size()[2]*self.unit)]+[int(x.size()[2]*self.other_unit) for i in range(self.multies-2)]+[x.size()[2]-int(x.size()[2]*self.unit)-(self.multies-2)*int(x.size()[2]*self.other_unit)],dim=2)
        
        # This processing logic is UNCHANGED
        # (it just calls the new models)
        x_list = [self.models[i](x_list[i]) for i in range(self.multies)]
        
        # This concatenation logic is UNCHANGED
        x = torch.cat(x_list,dim=2)
        
        # Pass to the TopModel as before
        x = self.top(x)
        return x
        
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)