import torch
from torch.utils.flop_counter import FlopCounterMode

from bdjscc import BDJSCC as model

def get_flops(model, input_size, with_backward=False):
    model.eval()
    dummy_input = torch.randn(input_size).cuda()
    flop_counter = FlopCounterMode(model)
    with flop_counter:
        if with_backward:
            model(dummy_input).sum().backward()
        else:
            with torch.no_grad():
                model(dummy_input)
    total_flops = flop_counter.get_total_flops()
    return total_flops

# Example usage:
input_size = (1, 3, 256, 256)  # Batch size 1, 3 color channels, 224x224 image
model = model().cuda()
flops = get_flops(model, input_size)
print(f"Total FLOPs: {flops}")