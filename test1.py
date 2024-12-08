import torch
import torch.nn as nn
from bdjscc import BDJSCC_Binary as model

# Load the model
model = model().cuda()

a = torch.randn(8, 3, 256, 256).cuda()

# Forward pass
output = model(a)
print(output.shape)