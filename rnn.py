# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

# # -----------------------------------------------------------------
# # BottomModel_ResNet50
# # -----------------------------------------------------------------
# class BottomModel_ResNet(nn.Module):
#     def __init__(self, gpu=False):
#         super(BottomModel_ResNet, self).__init__()
#         self.gpu = gpu

#         # Load ResNet-50 (training from scratch)
#         resnet = models.resnet50(weights=None)

#         # Use the initial convolution layers and layer1 only
#         self.features = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu,
#             resnet.maxpool,
#             resnet.layer1  # Output: 256 channels
#         )

#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         x = self.features(x)
#         return x


# # -----------------------------------------------------------------
# # TopModel (Updated for 256 channels from ResNet-50 layer1)
# # -----------------------------------------------------------------
# class TopModel(nn.Module):
#     def __init__(self, gpu=False, input_size=8):
#         super(TopModel, self).__init__()
#         self.gpu = gpu

#         # Input: 256 * 8 * 8
#         self.linear = nn.Linear(256 * input_size * 8, 256)
#         self.fc = nn.Linear(256, 256)
#         self.output = nn.Linear(256, 10)

#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()

#         B = x.size(0)
#         x = F.relu(self.linear(x.view(B, -1)))
#         x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
#         x = self.output(x)
#         return x


# # -----------------------------------------------------------------
# # Combined Model for Split Learning
# # -----------------------------------------------------------------
# class Model(nn.Module):
#     def __init__(self, gpu=False, multies=2, unit=0.25):
#         super(Model, self).__init__()
#         self.gpu = gpu
#         self.multies = multies
#         self.unit = unit
#         self.other_unit = (1 - unit) / (multies - 1)

#         # Create multiple ResNet bottom models
#         self.models = nn.ModuleList([BottomModel_ResNet(gpu) for _ in range(multies)])

#         # Height reduction by ResNet: 32 -> 16 -> 8
#         input_height = 32
#         output_height = input_height // 4

#         self.top = TopModel(gpu, input_size=output_height)

#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()

#         # Split input vertically among participants
#         splits = [int(x.size(2) * self.unit)] + [
#             int(x.size(2) * self.other_unit) for _ in range(self.multies - 2)
#         ]
#         splits.append(
#             x.size(2) - int(x.size(2) * self.unit) - (self.multies - 2) * int(x.size(2) * self.other_unit)
#         )
#         x_list = x.split(splits, dim=2)

#         # Each participant processes their chunk
#         x_list = [self.models[i](x_list[i]) for i in range(self.multies)]

#         # Concatenate processed features
#         x = torch.cat(x_list, dim=2)

#         # Top model for classification
#         x = self.top(x)
#         return x

#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ----------------------------- #
#     Feature Extractor Part    #
# ----------------------------- #
class FeatureExtractor(nn.Module):
    def __init__(self, gpu=False):
        super(FeatureExtractor, self).__init__()
        self.gpu = gpu

        # Load ResNet-50 pretrained model
        base_model = models.resnet50(pretrained=True)

        # Remove fully connected and average pool layers
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.features(x)
        return x


# ----------------------------- #
#     Top Model (Classifier)    #
# ----------------------------- #
class TopModel(nn.Module):
    def __init__(self, gpu=False, num_classes=3):
        super(TopModel, self).__init__()
        self.gpu = gpu

        # We will detect input size dynamically later
        self.linear = None
        self.fc = None
        self.output = None

        # Store shape after first forward pass
        self.initialized = False
        self.num_classes = num_classes

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size(0)

        # Detect input dimension once
        if not self.initialized:
            flattened_dim = x.view(B, -1).size(1)
            print(f"[INFO] Detected flattened feature dimension: {flattened_dim}")

            # Now define layers dynamically
            self.linear = nn.Linear(flattened_dim, 256)
            self.fc = nn.Linear(256, 256)
            self.output = nn.Linear(256, self.num_classes)

            if self.gpu:
                self.linear.cuda()
                self.fc.cuda()
                self.output.cuda()

            self.initialized = True

        x = F.relu(self.linear(x.view(B, -1)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)
        return x


# ----------------------------- #
#      Combined Full Model      #
# ----------------------------- #
class Model(nn.Module):
    def __init__(self, gpu=False, multies=2, unit=0.25):
        super(Model, self).__init__()
        self.gpu = gpu
        self.multies = multies
        self.unit = unit

        # Feature extractor (ResNet-50 conv layers)
        self.bottom = FeatureExtractor(gpu=gpu)

        # Classifier head (initialized dynamically)
        self.top = TopModel(gpu=gpu, num_classes=3)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.bottom(x)
        x = self.top(x)
        return x

