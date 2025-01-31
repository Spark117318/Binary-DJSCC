import torch
import torchvision.transforms as transforms
from bdjscc import BDJSCC as model
from train import store_test_image
from PIL import Image


# Load the model
model = model().cuda()

if False:

    # Load the image
    image = Image.open('/data/Users/lanli/ReActNet-master/dataset/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG')
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transforms(image).unsqueeze(0).cuda()

    # Load checkpoint
    checkpoint = torch.load('checkpoints/checkpoint_ada_thick_rprelu_6+_2.tar')
    model.load_state_dict(checkpoint['model'])

    # Forward pass
    output = model(image)

else:
    a = torch.zeros(1, 3, 256, 256).cuda()
    output = model(a)

print(output.shape)

# Store the test image
# store_test_image(image, output, -1, 0)
