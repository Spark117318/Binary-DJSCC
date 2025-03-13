import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from bdjscc_ import BDJSCC as model  # [bdjscc.py](bdjscc.py)
from train import store_test_image           # [train.py](train.py)
import os

def compute_psnr(mse):
    return 10 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))

if __name__ == "__main__":

    torch.cuda.set_device(2)

    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)

    # Create model
    model = model(channel_type='awgn').cuda()

    # Optionally load a checkpoint if available
    checkpoint = torch.load('checkpoints/checkpoint_thick_real.tar', map_location='cuda:2')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Dataloader
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    val_dataset = torchvision.datasets.ImageFolder('/data/Users/lanli/ReActNet-master/dataset/imagenet/val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True, generator=g)

    # Get first batch
    images, _ = next(iter(val_loader))
    images = images.cuda()

    snr_values = range(0, 21, 2)
    psnrs = []

    criterion = torch.nn.MSELoss()

    for snr in snr_values:
        model.snr_db = snr
        with torch.no_grad():
            output = model(images)
        mse = criterion(output, images).item()
        psnrs.append(compute_psnr(mse).item())

        # Store a sample image from this batch
        store_test_image(images[0].unsqueeze(0), output[0].unsqueeze(0), snr, 0)

    # Plot the SNR-PSNR chart
    plt.plot(snr_values, psnrs, marker='o')
    print(snr_values)
    print(psnrs)
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('BDJSCC_ada Performance vs. SNR')
    plt.grid(True)
    
    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/performance_vs_snr.png')