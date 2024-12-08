import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torchvision
import PIL

from bdjscc import BDJSCC_Binary as model

writer = SummaryWriter()

# Save the model
def save_model(model, optimizer, epoch, loss, filename):
    save_dict = {
        'model': model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(save_dict, filename)


def store_test_image(input, output, epoch, i):
    images_in = input[:9]
    images_out = output[:9]

    grid_in = torchvision.utils.make_grid(images_in, nrow=3, normalize=True)
    grid_out = torchvision.utils.make_grid(images_out, nrow=3, normalize=True)

    # images_in = input.cpu().detach().numpy()
    # images_out = output.cpu().detach().numpy()

    # writer.add_image('Test Image', grid, epoch)

    # Convert to PIL image
    ndarr_in = grid_in.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im_in = PIL.Image.fromarray(ndarr_in)
    ndarr_out = grid_out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im_out = PIL.Image.fromarray(ndarr_out)

    images_path = os.path.join(os.getcwd(), 'images')
    os.makedirs(images_path, exist_ok=True)

    im_in.save(os.path.join(images_path, f'test_image_in_{epoch}_{i}.png'))
    im_out.save(os.path.join(images_path, f'test_image_out_{epoch}_{i}.png'))



if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    epochs = 20
    learning_rate = 1e-4

    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_tar = os.path.join(checkpoint_path, 'checkpoint.tar')


    # Load the model
    model = model().cuda()

    # Load the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load the loss function
    criterion = nn.MSELoss()

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder('../data/imagenet/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model(autoencoder)
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['model'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f"Resuming training from epoch {start_epoch} with loss {loss}")

    else:
        start_epoch = 0
        loss = 0

    for epoch in range(start_epoch, epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images.cuda()

            # Forward pass
            output = model(images)

            # Compute the loss
            loss = criterion(output, images)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")

            if i % 100 == 0:
                save_model(model, optimizer, epoch, loss, checkpoint_tar)
                store_test_image(images, output, epoch, i)

            writer.add_scalar('Loss/train1', loss.item(), epoch * len(train_loader) + i)
