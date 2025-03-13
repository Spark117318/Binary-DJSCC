import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torchvision
import PIL
from PIL import Image
from ABNet import adaptive_clip_grad

from bdjscc_ import BDJSCC_ada as model
# from ABdjscc import BDJSCC_AB as model

writer = SummaryWriter()

torch.cuda.set_device(2)
# torch.cuda.set_per_process_memory_fraction(0.3, 2)

# Save the model
def save_model(model, optimizer, epoch, loss, filename):
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(save_dict, filename)


def store_test_image(input, output, epoch, i):
    images_in = input[:9]
    images_out = output[:9]

    nrow = int(images_in.shape[0] ** 0.5)

    grid_in = torchvision.utils.make_grid(images_in, nrow=nrow, normalize=True)
    grid_out = torchvision.utils.make_grid(images_out, nrow=nrow, normalize=True)

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

    im_in.save(os.path.join(images_path, f'test_image_{epoch}_{i}_in.png'))
    im_out.save(os.path.join(images_path, f'test_image_{epoch}_{i}_out.png'))



if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    epochs = 10
    learning_rate = 1e-3
    weight_decay = 0

    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_tar = os.path.join(checkpoint_path, 'checkpoint_react_thick_qrprelu_omni_binary_1.tar')
    checkpoint_tar_store = os.path.join(checkpoint_path, 'checkpoint_react_thick_qrprelu_omni_binary_1.tar')

    # checkpoint_tar = os.path.join(checkpoint_path, 'checkpoint_ada_thick_rprelu.tar')
    # checkpoint_tar_store = os.path.join(checkpoint_path, 'checkpoint_ada_thick_rprelu_omini-.tar')

    # Load the model
    model = model().cuda()


    # all_parameters = model.parameters()
    # weight_parameters = []
    # for pname, p in model.named_parameters():
    #     if p.ndimension() == 4 or 'conv' in pname:
    #         weight_parameters.append(p)
    # weight_parameters_id = list(map(id, weight_parameters))
    # other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # optimizer = torch.optim.Adam(
    #         [{'params' : other_parameters},
    #         {'params' : weight_parameters, 'weight_decay' : weight_decay}],
    #         lr=learning_rate,
    #         betas=(0.9,0.999))


    # Load the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*10, eta_min=1e-5)

    # Load the loss function
    criterion = nn.MSELoss()

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder('/data/Users/lanli/ReActNet-master/dataset/imagenet/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    # Load the test image
    # image_test = Image.open('/data/Users/lanli/ReActNet-master/dataset/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG')
    # transforms = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])
    # image_test = transforms(image_test).unsqueeze(0).cuda()


    # Train the model(autoencoder)
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar, map_location='cuda:2')
        model.load_state_dict(checkpoint['model'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f"Resuming training from epoch {start_epoch} with loss {loss}")

    else:
        loss = 1

    start_epoch = 0
    loss_best = 1
    for epoch in range(start_epoch, epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images.cuda()

            # Forward pass
            output = model(images)

            # Crop the input and output
            # images = images[:, :, 4:252, 4:252]
            # output = output[:, :, 4:252, 4:252]

            # Compute the loss
            loss = criterion(output, images)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Clip the gradients
            # adaptive_clip_grad(model.parameters())

            # Update the weights
            optimizer.step()

            if i % 10 == 0 and i != 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")
                mse = loss.item()
                psnr = 10 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
                print(f"PSNR: {psnr}")
                # Test the model
                # model.eval()
                # with torch.no_grad():
                #     output_test = model(image_test)
                #     loss_test = criterion(output_test[:, :, 4:252, 4:252], image_test[:, :, 4:252, 4:252])
                #     print(f"Test Loss: {loss_test.item()}")
                # model.train()


            if i % 100 == 0 and i != 0:
                if loss.item() < loss_best:
                    loss_best = loss.item()
                    print(f"Saving the model with loss {loss_best}")
                    save_model(model, optimizer, epoch, loss, checkpoint_tar_store)
                    # store_test_image(images, output, epoch, i)
                # lr_scheduler.step()

            writer.add_scalar('Loss/train1', loss.item(), epoch * len(train_loader) + i)
