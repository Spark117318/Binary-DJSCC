import torch
import torchvision.transforms as transforms
from bdjscc import BDJSCC as model
from train import store_test_image
from PIL import Image


# Load the model
model = model().cuda()

if False:
    # Load the image
    image = Image.open('/home/Spark/learning/research/semantic_comm/data/imagenet/train/n01795545/n01795545_10144.JPEG')
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transforms(image).unsqueeze(0).cuda()

    # Load checkpoint
    checkpoint = torch.load('checkpoints/checkpoint.tar')
    model.load_state_dict(checkpoint['model'])

    # Forward pass
    output = model(image)

else:
    a = torch.zeros(9, 3, 256, 256).cuda()+0.5
    output = model(a)

print(output.shape)

# Store the test image
# store_test_image(image, output, -1, 0)
