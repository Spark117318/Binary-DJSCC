import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from bdjscc import BDJSCC_ada as model
from train import store_test_image

torch.cuda.set_device(1)

batch_size = 256

model = model().cuda()
model = model.eval()

checkpoint = torch.load('checkpoints/checkpoint_ada_thick_rprelu_6+.tar')
model.load_state_dict(checkpoint['model'])

criteria = torch.nn.MSELoss()
loss = 0

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('/data/Users/lanli/ReActNet-master/dataset/imagenet/val', transform=val_transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True
)

print(len(val_loader))

for i, (images, _) in enumerate(val_loader):
    images = images.cuda()

    with torch.no_grad():
        output = model(images)

    loss += criteria(output, images).item()

    if i % 100 == 0:
        loss_ = round(criteria(output[1], images[1]).item(), 6)
        store_test_image(images[1], output[1], i, loss_)
        # print(f"Progress: {i}/{1000}")

    # if i == 1000:
    #     break

loss_mean = loss/len(val_loader)
print(f"Validation loss: {loss_mean}")
psnr = 10 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(loss_mean))
print(f"PSNR: {psnr}")
    
