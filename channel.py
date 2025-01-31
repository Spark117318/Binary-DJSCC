import torch
import torch.nn as nn
import math

class Channel(nn.Module):
    """
    A PyTorch module that simulates different wireless channels (AWGN, slow-fading, slow-fading
    with equalization, burst noise, etc.) similar to the TensorFlow implementation you provided.
    
    For convenience, the random SNR (in dB) is automatically chosen in [1, 30] if `snr_db` is None.
    All operations are done on the same device as the input `features`.
    """

    def __init__(self, channel_type='awgn'):
        """
        channel_type (str): One of ['awgn', 'slow_fading', 'slow_fading_eq', 'burst'].
        """
        super(Channel, self).__init__()
        self.channel_type = channel_type

    def forward(self, features, snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        """
        features:  Tensor of shape [batch_size, ..., 2*dim_z], where the last dimension 
                   holds real and imaginary parts concatenated.
        snr_db:    Tensor or float indicating the SNR in dB; if None, a random SNR from [1..30].
        h_real:    Real part of channel coefficient (for slow fading).
        h_imag:    Imag part of channel coefficient (for slow fading).
        b_prob:    Probability of burst occurrence (for burst noise).
        b_stddev:  Stddev multiplier for the burst noise amplitude (for burst noise).
        """
        # Flatten all but batch dimension so we can treat it like [batch_size, 2*dim_z]
        batch_size = features.shape[0]
        inter_shape = features.shape
        f = features.view(batch_size, -1)

        # Split real & imaginary
        dim_z = f.shape[1] // 2
        z_real = f[:, :dim_z]
        z_imag = f[:, dim_z:]
        
        # Construct complex tensor
        z_in = torch.complex(z_real, z_imag)

        # Power constraint: ensure average complex symbol power = 1
        #   norm_factor = sum(|z_in|^2) over each sample
        #   multiply z_in by sqrt(dim_z / norm_factor)
        norm_factor = (z_in.real**2 + z_in.imag**2).sum(dim=1, keepdim=True)  # shape: [batch_size, 1]
        # Avoid divide-by-zero if norm_factor is extremely small
        norm_factor = torch.clamp(norm_factor, min=1e-12)
        z_in_norm = z_in * torch.sqrt(torch.tensor(dim_z, dtype=z_in.dtype, device=z_in.device) / norm_factor)

        # If no SNR provided, pick a random integer [1..30] for each batch or for entire batch
        # If you'd like a single random SNR per batch, remove the "batch_size," shape below.
        if snr_db is None:
            snr_db = torch.randint(low=1, high=20, size=(batch_size,), device=z_in.device)
        elif isinstance(snr_db, float) or isinstance(snr_db, int):
            # expand to match batch size
            snr_db = torch.full((batch_size,), int(snr_db), device=z_in.device)
        # At this point, snr_db is a tensor of shape [batch_size]
        # print(snr_db)

        # Depending on the channel type, apply the appropriate transform
        if self.channel_type == 'awgn':
            z_out = self.awgn(z_in_norm, snr_db)
        elif self.channel_type == 'slow_fading':
            if h_real is None or h_imag is None:
                raise ValueError("For 'slow_fading', both h_real and h_imag must be provided.")
            z_out = self.slow_fading(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'slow_fading_eq':
            if h_real is None or h_imag is None:
                raise ValueError("For 'slow_fading_eq', both h_real and h_imag must be provided.")
            z_out = self.slow_fading_eq(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'burst':
            if b_prob is None or b_stddev is None:
                raise ValueError("For 'burst', b_prob and b_stddev must be provided.")
            z_out = self.burst(z_in_norm, snr_db, b_prob, b_stddev)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        # Convert z_out (complex) back to real and imaginary
        z_out_real = z_out.real
        z_out_imag = z_out.imag
        
        # Concatenate and reshape back to original
        z_out_concat = torch.cat([z_out_real, z_out_imag], dim=1)
        z_out_concat = z_out_concat.view(*inter_shape)

        return z_out_concat

    def awgn(self, x, snr_db):
        """
        Add AWGN to x given random or supplied SNR values in dB.
        noise_stddev = sqrt(10^(-snr_db/10)).
        
        x:       complex tensor, shape [batch_size, dim_z]
        snr_db:  tensor of shape [batch_size]
        """
        # noise_stddev is per sample, so expand dims for broadcast if needed
        noise_stddev = torch.pow(10.0, -snr_db / 10.0)  # 10^(-snr_dB/20)
        # shape [batch_size, 1] for broadcasting
        noise_stddev = noise_stddev.view(-1, 1)

        # Generate complex Gaussian noise with variance 1 per dimension => std = 1/sqrt(2)
        noise_real = torch.randn_like(x.real) / math.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / math.sqrt(2.0)
        noise = torch.complex(noise_real, noise_imag)

        # Scale noise by noise_stddev
        # (broadcast along dim=1, so same noise scaling across x's second dimension)
        return x + noise_stddev * noise

    def slow_fading(self, x, snr_db, h_real, h_imag):
        """
        y = h*x + noise, with h as the same for entire codeword (per sample).
        """
        noise_stddev = torch.pow(10.0, -snr_db / 20.0).view(-1, 1)
        
        # reshape h to [batch_size, 1]
        h = torch.complex(h_real, h_imag).view(-1, 1)
        
        noise_real = torch.randn_like(x.real) / math.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / math.sqrt(2.0)
        awgn = torch.complex(noise_real, noise_imag)

        return h * x + noise_stddev * awgn

    def slow_fading_eq(self, x, snr_db, h_real, h_imag):
        """
        y = x + noise/h, i.e. equalization at receiver side for slow-fading channel.
        """
        noise_stddev = torch.pow(10.0, -snr_db / 20.0).view(-1, 1)

        h = torch.complex(h_real, h_imag).view(-1, 1)
        
        noise_real = torch.randn_like(x.real) / math.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / math.sqrt(2.0)
        awgn = torch.complex(noise_real, noise_imag)

        return x + (noise_stddev * awgn) / h

    def burst(self, x, snr_db, b_prob, b_stddev):
        """
        y = x + AWGN + burst_noise, where burst occurs with probability b_prob,
        and burst amplitude is scaled by b_stddev.
        """
        # AWGN
        noise_stddev = torch.pow(10.0, -snr_db / 20.0).view(-1, 1)
        noise_real = torch.randn_like(x.real) / math.sqrt(2.0)
        noise_imag = torch.randn_like(x.imag) / math.sqrt(2.0)
        awgn = torch.complex(noise_real, noise_imag)

        # Burst noise
        #   sample = 1 with prob b_prob, else 0
        #   scale by b_stddev
        #   draw independent noise for the burst
        # Ensure b_prob, b_stddev are Tensors on correct device
        if not isinstance(b_prob, torch.Tensor):
            b_prob = torch.tensor(b_prob, dtype=x.dtype, device=x.device)
        if not isinstance(b_stddev, torch.Tensor):
            b_stddev = torch.tensor(b_stddev, dtype=x.dtype, device=x.device)

        # Same shape as x.real
        # For a per-element burst, do:
        #   b_sample = torch.bernoulli(b_prob * torch.ones_like(x.real))
        # If you want a single random burst event per sample (not each element),
        # you can do:
        #   b_sample = torch.bernoulli(b_prob.expand(x.size(0))).view(-1, 1)
        b_sample = torch.bernoulli(b_prob * torch.ones_like(x.real))
        b_sample = torch.complex(b_sample, torch.zeros_like(b_sample))

        burst_real = torch.randn_like(x.real) / math.sqrt(2.0)
        burst_imag = torch.randn_like(x.imag) / math.sqrt(2.0)
        burst_noise = torch.complex(burst_real, burst_imag) * b_stddev * b_sample

        return x + noise_stddev * awgn + burst_noise
