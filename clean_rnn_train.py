# cnn_model_multi.py  -- ResNet-like bottom model (no torchvision dependency)
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --- small residual block ---
# class BasicBlockSimple(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlockSimple, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = F.relu(out)
#         return out

# # --- Bottom model: small ResNet-like stem and layer1 ---
# class BottomModel(nn.Module):
#     def __init__(self, gpu=False):
#         super(BottomModel, self).__init__()
#         self.gpu = gpu
#         # CIFAR-style stem
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         # two basic blocks -> keeps output channels=64
#         self.layer1 = nn.Sequential(
#             BasicBlockSimple(64, 64, stride=1),
#             BasicBlockSimple(64, 64, stride=1)
#         )
#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         return x

# # --- TopModel unchanged from your original ---
# class TopModel(nn.Module):
#     def __init__(self,gpu=False,input_size=8):
#         super(TopModel,self).__init__()
#         self.gpu = gpu
#         self.linear = nn.Linear(64*input_size*8, 256)
#         self.fc = nn.Linear(256, 256)
#         self.output = nn.Linear(256, 10)
#         if gpu:
#             self.cuda()

#     def forward(self,x):
#         if self.gpu:
#             x = x.cuda()
#         B = x.size()[0]
#         x = F.relu(self.linear(x.view(B,-1)))
#         x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
#         x = self.output(x)
#         return x

# # --- Full split Model: multiple bottom parts + top ---
# class Model(nn.Module):
#     def __init__(self, gpu=False,multies=2,unit = 0.25):
#         super(Model, self).__init__()
#         self.gpu = gpu
#         self.multies = multies
#         self.unit = unit
#         self.other_unit = (1-unit)/(multies-1)
#         self.models = nn.ModuleList([BottomModel(gpu) for i in range(self.multies)])
#         # Use expected output height: for 32x32 input, bottom reduces by approx 4x in height
#         input_height = 32
#         output_height = input_height // 4
#         self.top = TopModel(gpu, int(output_height))
#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         x_list = x.split(
#             [int(x.size()[2]*self.unit)] + [int(x.size()[2]*self.other_unit) for i in range(self.multies-2)] +
#             [x.size()[2]-int(x.size()[2]*self.unit)-(self.multies-2)*int(x.size()[2]*self.other_unit)],
#             dim=2)
#         x_list = [self.models[i](x_list[i]) for i in range(self.multies)]
#         x = torch.cat(x_list, dim=2)
#         x = self.top(x)
#         return x

#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)


# # clean_rnn_train.py
# import argparse
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import time
# from rnn import Model  # ✅ Import your ResNet-based model

# # --------------------------------------------------------------------
# # 1️⃣ Argument parsing
# # --------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--clean-epoch', type=int, default=80, help='number of clean epochs')
# parser.add_argument('--dup', type=int, default=0, help='duplicate index')
# parser.add_argument('--multies', type=int, default=4, help='number of model partitions')
# parser.add_argument('--unit', type=float, default=0.25, help='split unit ratio')
# args = parser.parse_args()

# # --------------------------------------------------------------------
# # 2️⃣ Settings
# # --------------------------------------------------------------------
# GPU = False   # ❗ change to True if you have CUDA and GPU torch installed
# batch_size = 128
# lr = 0.001
# EPOCHS = args.clean_epoch

# device = torch.device('cuda' if GPU and torch.cuda.is_available() else 'cpu')
# print(f"Training on device: {device}")

# # --------------------------------------------------------------------
# # 3️⃣ Prepare dataset (CIFAR-10)
# # --------------------------------------------------------------------
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, 4),
#     transforms.ToTensor(),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])

# train_dataset = datasets.CIFAR10(root='./raw_data', train=True, download=True, transform=transform_train)
# test_dataset = datasets.CIFAR10(root='./raw_data', train=False, download=True, transform=transform_test)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# # --------------------------------------------------------------------
# # 4️⃣ Initialize model and optimizer
# # --------------------------------------------------------------------
# model = Model(gpu=GPU, multies=args.multies, unit=args.unit).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# # --------------------------------------------------------------------
# # 5️⃣ Training function
# # --------------------------------------------------------------------
# def train(epoch):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = F.cross_entropy(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     avg_loss = running_loss / total
#     acc = 100. * correct / total
#     print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

# # --------------------------------------------------------------------
# # 6️⃣ Testing function
# # --------------------------------------------------------------------
# def test():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#     acc = 100. * correct / total
#     return acc

# # --------------------------------------------------------------------
# # 7️⃣ Train the model
# # --------------------------------------------------------------------
# start_time = time.time()

# for epoch in range(EPOCHS):
#     train(epoch)

# train_time = time.time() - start_time

# # --------------------------------------------------------------------
# # 8️⃣ Evaluate and save
# # --------------------------------------------------------------------
# test_acc = test()
# print(f"✅ Clean model test accuracy: {test_acc:.2f}%")
# print(f"🕒 Training time: {train_time:.2f} seconds")

# save_path = f"clean_rnn_{args.dup}_{args.multies}_{args.unit}.model"
# torch.save(model.state_dict(), save_path)
# print(f"💾 Model saved to: {save_path}")


# import torch
# import torchvision
# import torchvision.transforms as transforms
# import argparse
# import time
# from resnet_model_multi import Model   # <-- changed import

# parser = argparse.ArgumentParser()
# parser.add_argument('--clean-epoch', type=int, required=False, default=80)
# parser.add_argument('--dup', type=int, required=True)
# parser.add_argument('--multies', type=int, required=False, default=2)
# parser.add_argument('--unit', type=float, required=False, default=0.25)

# def train_model(model, dataloader, epoch_num, is_binary, verbose=True):
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     for epoch in range(epoch_num):
#         cum_loss, cum_acc, tot = 0, 0, 0
#         for x_in, y_in in dataloader:
#             B = x_in.size(0)
#             pred = model(x_in)
#             loss = model.loss(pred, y_in)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             cum_loss += loss.item() * B
#             pred_c = pred.max(1)[1].cpu()
#             cum_acc += (pred_c.eq(y_in)).sum().item()
#             tot += B
#         if verbose:
#             print(f"Epoch {epoch+1}/{epoch_num}, loss={cum_loss/tot:.4f}, acc={cum_acc/tot:.4f}")

# def eval_model(model, dataloader, is_binary):
#     model.eval()
#     cum_acc, tot = 0, 0
#     for x_in, y_in in dataloader:
#         pred = model(x_in)
#         pred_c = pred.max(1)[1].cpu()
#         cum_acc += (pred_c.eq(y_in)).sum().item()
#         tot += x_in.size(0)
#     return cum_acc / tot

# if __name__ == "__main__":
#     args = parser.parse_args()
#     GPU = torch.cuda.is_available()
#     BATCH_SIZE = 500

#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=5),
#         transforms.RandomRotation(10),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465],
#                              [0.247, 0.243, 0.261])
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465],
#                              [0.247, 0.243, 0.261])
#     ])

#     trainset = torchvision.datasets.CIFAR10(root='./raw_data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR10(root='./raw_data', train=False, download=True, transform=transform_test)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     model = Model(gpu=GPU, multies=args.multies, unit=args.unit)

#     start = time.time()
#     train_model(model, trainloader, epoch_num=args.clean_epoch, is_binary=False)
#     acc = eval_model(model, testloader, is_binary=False)
#     torch.save(model.state_dict(), f"clean_resnet-{args.dup}-{args.multies}-{args.unit}.model")
#     print(f"✅ Clean ResNet model acc: {acc:.4f}")
#     print(f"Training time: {time.time() - start:.2f}s")


# clean_train_resnet.py
# """
# Train a clean ResNet-based split model (uses rnn.py -> Model).
# Safe for Windows (multiprocessing guard) and works with CPU/GPU.
# """

# import argparse
# import time
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms

# # import your ResNet-based Model (rnn.py)
# from rnn import Model

# parser = argparse.ArgumentParser()
# parser.add_argument('--clean-epoch', type=int, default=80, help='the number of training epochs without poisoning')
# parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
# parser.add_argument('--multies', type=int, default=2, help='the number of multiple participants')
# parser.add_argument('--unit', type=float, default=0.25, help='the feature ratio held by the attacker')
# parser.add_argument('--batch-size', type=int, default=128, help='batch size (reduce for CPU)')
# parser.add_argument('--num-workers', type=int, default=0, help='dataloader num_workers (0 for Windows/CPU)')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

# def train_model(model, dataloader, optimizer, epoch_num, is_binary=False, verbose=True):
#     model.train()
#     for epoch in range(epoch_num):
#         cum_loss = 0.0
#         cum_acc = 0.0
#         tot = 0
#         for i, (x_in, y_in) in enumerate(dataloader):
#             x_in = x_in.to(device)
#             y_in = y_in.to(device)
#             pred = model(x_in)
#             loss = F.cross_entropy(pred, y_in)  # use cross_entropy directly
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             B = x_in.size(0)
#             cum_loss += loss.item() * B
#             pred_c = pred.max(1)[1].cpu()
#             cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
#             tot += B
#         if verbose:
#             print(f"Epoch {epoch}, loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
#     return

# def eval_model(model, dataloader, is_binary=False):
#     model.eval()
#     cum_acc = 0
#     tot = 0
#     with torch.no_grad():
#         for i, (x_in, y_in) in enumerate(dataloader):
#             x_in = x_in.to(device)
#             y_in = y_in.to(device)
#             pred = model(x_in)
#             pred_c = pred.max(1)[1].cpu()
#             cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
#             tot += x_in.size(0)
#     return cum_acc / tot

# if __name__ == '__main__':
#     args = parser.parse_args()

#     # Decide device
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')
#     print("Training on device:", device)

#     # Reproducibility seeds
#     torch.manual_seed(0)
#     if use_cuda:
#         torch.cuda.manual_seed_all(0)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     BATCH_SIZE = args.batch_size
#     N_EPOCH = 100
#     CLEAN_EPOCH = args.clean_epoch

#     # Data transforms (same normalization as your project)
#     transform_for_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=5),
#         transforms.RandomRotation(10),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465],
#                              [0.2470, 0.2430, 0.2610])
#     ])
#     transform_for_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465],
#                              [0.2470, 0.2430, 0.2610])
#     ])

#     # Datasets (will download if needed)
#     trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform_for_train)
#     testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True, transform=transform_for_test)

#     trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

#     # Create model (pass gpu flag so rnn.py internal checks don't force cuda incorrectly)
#     model = Model(gpu=use_cuda, multies=args.multies, unit=args.unit)
#     # Ensure model on proper device (if rnn.py internally used .cuda, it's ok; calling to(device) is safe)
#     model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # Train clean epochs first (CLEAN_EPOCH)
#     t1 = time.time()
#     if CLEAN_EPOCH > 0:
#         print(f"Starting {CLEAN_EPOCH} clean epochs...")
#         train_model(model, trainloader, optimizer, epoch_num=CLEAN_EPOCH, is_binary=False, verbose=True)
#         torch.save(model.state_dict(), f'clean_epoch_{args.dup}-{args.multies}-{args.unit}.model')

#     # Continue to full N_EPOCH
#     remaining = N_EPOCH - CLEAN_EPOCH
#     if remaining > 0:
#         print(f"Continuing training for remaining {remaining} epochs...")
#         train_model(model, trainloader, optimizer, epoch_num=remaining, is_binary=False, verbose=True)

#     cleanacc = eval_model(model, testloader, is_binary=False)
#     torch.save(model.state_dict(), f'clean-{args.dup}-{args.multies}-{args.unit}.model')
#     print('clean acc: %.4f' % cleanacc)
#     t2 = time.time()
#     print("Training a model costs %.4fs." % (t2 - t1))


# import argparse
# import os
# import time
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from rnn import Model  # your ResNet-based model

# # ----------------------------- #
# #        Argument Parser        #
# # ----------------------------- #
# parser = argparse.ArgumentParser()
# parser.add_argument('--clean-epoch', type=int, default=80, help='the number of clean training epochs')
# parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
# parser.add_argument('--multies', type=int, default=2, help='the number of multiple participants')
# parser.add_argument('--unit', type=float, default=0.25, help='the feature ratio held by the attacker')
# parser.add_argument('--batch-size', type=int, default=32, help='batch size (lower for CPU)')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# parser.add_argument('--data-dir', type=str, default=r"C:\Users\aront\Downloads\dataset\dataset", help='path to dataset folder')

# # ----------------------------- #
# #       Train & Eval Loop       #
# # ----------------------------- #
# def train_model(model, dataloader, optimizer, epoch_num, device):
#     model.train()
#     for epoch in range(epoch_num):
#         total_loss, total_correct, total_samples = 0, 0, 0
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = F.cross_entropy(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * inputs.size(0)
#             preds = outputs.argmax(dim=1)
#             total_correct += preds.eq(labels).sum().item()
#             total_samples += inputs.size(0)

#         print(f"Epoch [{epoch+1}/{epoch_num}] | Loss: {total_loss/total_samples:.4f} | "
#               f"Acc: {total_correct/total_samples:.4f}")

# def eval_model(model, dataloader, device):
#     model.eval()
#     total_correct, total_samples = 0, 0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             preds = outputs.argmax(dim=1)
#             total_correct += preds.eq(labels).sum().item()
#             total_samples += inputs.size(0)
#     return total_correct / total_samples

# # ----------------------------- #
# #          Main Script          #
# # ----------------------------- #
# if __name__ == '__main__':
#     args = parser.parse_args()

#     # Device setup
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')
#     print(f"Training on device: {device}")

#     # Set seeds for reproducibility
#     torch.manual_seed(0)
#     if use_cuda:
#         torch.cuda.manual_seed_all(0)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     # -------------------- #
#     #  Dataset Definition  #
#     # -------------------- #
#     transform_train = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     dataset = ImageFolder(root=args.data_dir, transform=transform_train)
#     num_classes = len(dataset.classes)
#     print(f"Detected classes: {dataset.classes}")

#     # Split dataset into 80% train, 20% test
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     # -------------------- #
#     #    Model Creation    #
#     # -------------------- #
#     model = Model(gpu=use_cuda, multies=args.multies, unit=args.unit)
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # -------------------- #
#     #     Training Loop    #
#     # -------------------- #
#     start_time = time.time()
#     print(f"Starting {args.clean_epoch} clean epochs...")
#     train_model(model, train_loader, optimizer, args.clean_epoch, device)

#     print("Evaluating model...")
#     acc = eval_model(model, test_loader, device)
#     print(f"Clean Accuracy: {acc:.4f}")

#     torch.save(model.state_dict(), f"clean_resnet_{args.dup}-{args.multies}-{args.unit}.pth")
#     print(f"Model saved as clean_resnet_{args.dup}-{args.multies}-{args.unit}.pth")

#     print(f"Training finished in {time.time() - start_time:.2f}s.")

# import argparse
# import os
# import time
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from rnn import Model  # ResNet-50 model

# # ----------------------------- #
# #        Argument Parser        #
# # ----------------------------- #
# parser = argparse.ArgumentParser()
# parser.add_argument('--clean-epoch', type=int, default=80)
# parser.add_argument('--dup', type=int, required=True)
# parser.add_argument('--multies', type=int, default=2)
# parser.add_argument('--unit', type=float, default=0.25)
# parser.add_argument('--batch-size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--data-dir', type=str,
#                     default=r"C:\Users\aront\Downloads\dataset\dataset")

# # ----------------------------- #
# #       Train & Eval Loop       #
# # ----------------------------- #
# def train_model(model, dataloader, optimizer, epoch_num, device):
#     model.train()
#     for epoch in range(epoch_num):
#         total_loss, total_correct, total_samples = 0, 0, 0
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = F.cross_entropy(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * inputs.size(0)
#             preds = outputs.argmax(dim=1)
#             total_correct += preds.eq(labels).sum().item()
#             total_samples += inputs.size(0)

#         print(f"Epoch [{epoch + 1}/{epoch_num}] | "
#               f"Loss: {total_loss / total_samples:.4f} | "
#               f"Acc: {total_correct / total_samples:.4f}")

# def eval_and_plot(model, dataloader, device, class_names):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             preds = outputs.argmax(dim=1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())

#     acc = np.mean(np.array(y_true) == np.array(y_pred))
#     print(f"\nOverall Accuracy: {acc:.4f}\n")
#     print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     plt.figure(figsize=(7, 6))
#     sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel("Predicted Labels", fontsize=12)
#     plt.ylabel("True Labels", fontsize=12)
#     plt.title("Confusion Matrix – ResNet50 RF Signal Classification", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("confusion_matrix_resnet50.png", dpi=300)
#     plt.show()

#     print("\nConfusion matrix saved as 'confusion_matrix_resnet50.png'")
#     return acc

# # ----------------------------- #
# #          Main Script          #
# # ----------------------------- #
# if __name__ == '__main__':
#     args = parser.parse_args()

#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')
#     print(f"Training on device: {device}")

#     torch.manual_seed(0)
#     if use_cuda:
#         torch.cuda.manual_seed_all(0)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     # -------------------- #
#     #  Dataset Definition  #
#     # -------------------- #
#     transform_train = transforms.Compose([
#         transforms.Resize((180, 180)),  # match paper input shape
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])

#     dataset = ImageFolder(root=args.data_dir, transform=transform_train)
#     class_names = dataset.classes
#     num_classes = len(class_names)
#     print(f"Detected classes: {class_names}")

#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     # -------------------- #
#     #    Model Creation    #
#     # -------------------- #
#     model = Model(gpu=use_cuda, multies=args.multies, unit=args.unit)
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # -------------------- #
#     #     Training Loop    #
#     # -------------------- #
#     start_time = time.time()
#     print(f"Starting {args.clean_epoch} clean epochs...")
#     train_model(model, train_loader, optimizer, args.clean_epoch, device)

#     # -------------------- #
#     #    Evaluation Plot   #
#     # -------------------- #
#     print("Evaluating model and generating confusion matrix...")
#     acc = eval_and_plot(model, test_loader, device, class_names)

#     torch.save(model.state_dict(),
#                f"clean_resnet50_{args.dup}-{args.multies}-{args.unit}.pth")
#     print(f"Model saved as clean_resnet50_{args.dup}-{args.multies}-{args.unit}.pth")
#     print(f"Training finished in {time.time() - start_time:.2f}s.")


import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rnn import Model  # <-- keep your ResNet50 model as it is

# ----------------------------- #
#        Argument Parser        #
# ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--clean-epoch', type=int, default=80)
parser.add_argument('--dup', type=int, required=True)
parser.add_argument('--multies', type=int, default=2)
parser.add_argument('--unit', type=float, default=0.25)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data-dir', type=str,
                    default=r"C:\Users\aront\Downloads\dataset\dataset")

# ----------------------------- #
#      Train / Validate Loop    #
# ----------------------------- #
def train_and_validate(model, train_loader, val_loader, optimizer, epochs, device):
    """Trains and evaluates the model each epoch (train + validation)."""
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        # ----- Training -----
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += preds.eq(labels).sum().item()
            train_total += inputs.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # ----- Validation -----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += inputs.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        # ----- Print per epoch -----
        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ----------------------------- #
#     Evaluation (Confusion)    #
# ----------------------------- #
def evaluate_and_plot(model, test_loader, device, class_names):
    """Evaluates model on test data and generates confusion matrix."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n=== Final Test Accuracy: {acc:.4f} ===\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ----- Confusion Matrix -----
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix – ResNet50 RF Signal Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix_resnet50.png", dpi=300)
    plt.show()

    print("Confusion matrix saved as 'confusion_matrix_resnet50.png'")
    return acc

# ----------------------------- #
#          Main Script          #
# ----------------------------- #
if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Training on device: {device}")

    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------- #
    #  Dataset Definition  #
    # -------------------- #
    transform_all = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=args.data_dir, transform=transform_all)
    class_names = dataset.classes
    print(f"Detected classes: {class_names}")

    # Split data: 70% train, 15% val, 15% test
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # -------------------- #
    #    Model Creation    #
    # -------------------- #
    model = Model(gpu=use_cuda, multies=args.multies, unit=args.unit)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- #
    #     Train + Valid    #
    # -------------------- #
    start_time = time.time()
    print(f"\nStarting training for {args.clean_epoch} epochs...\n")
    train_and_validate(model, train_loader, val_loader, optimizer, args.clean_epoch, device)

    # -------------------- #
    #   Test & Confusion   #
    # -------------------- #
    print("\nEvaluating model on test data...\n")
    test_acc = evaluate_and_plot(model, test_loader, device, class_names)

    # Save model
    torch.save(model.state_dict(),
               f"clean_resnet50_{args.dup}-{args.multies}-{args.unit}.pth")
    print(f"Model saved as clean_resnet50_{args.dup}-{args.multies}-{args.unit}.pth")
    print(f"\nTraining finished in {time.time() - start_time:.2f}s.")
