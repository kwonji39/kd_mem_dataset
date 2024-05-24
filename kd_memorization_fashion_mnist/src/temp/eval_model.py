import sys
import torch
# Add the directory containing the 'resnet.py' file to the Python path
sys.path.append('/scratch/gilbreth/kwon165/kd_memorization/src/models')

# Now you can import the ResNet module
import resnet as ResNet

# You can now use ResNet in your code
model_t = ResNet.resnet50(num_classes=10)  # Assuming there's a ResNet class defined in resnet.py
# model_t.load_state_dict(torch.load('/scratch/gilbreth/kwon165/kd_memorization/src/temp/res50_20240404-232030')["model_state_dict"])
model_t.load_state_dict(torch.load('/scratch/gilbreth/kwon165/kd_memorization/src/temp/res50_20240405-144308/checkpoints/200/model.pth')["model_state_dict"])
model_t.eval()
# torch.save(model.state_dict(), '/scratch/gilbreth/kwon165/kd_memorization/src/temp/f_mnist_teacher.pth')

model_s = ResNet.resnet18(num_classes=10)
# model_s.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model_s.conv1 = torch.nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3, bias=False)

# model_s.load_state_dict(torch.load('/scratch/gilbreth/kwon165/kd_memorization/src/temp/res18_20240412-001159/checkpoints/1/model.pth')["model_state_dict"])
# model_s.eval()



# RuntimeError: Given groups=1,
# weight of size [64, 3, 3, 3], expected input[256, 1, 28, 28] to have 3 channels, but got 1 channels instead