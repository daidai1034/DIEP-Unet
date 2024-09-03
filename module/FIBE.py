import torch
import torch.nn as nn

class FIBE(nn.Module):
    def __init__(self):
        super(FIBE, self).__init__()

    def forward(self, x):
        spectrum = torch.fft.fftn(x, dim=(-2, -1))
        phase_spectrum = torch.angle(spectrum)
        f_reconstruct = torch.exp(1j * phase_spectrum)
        tran_phase = torch.real(torch.fft.ifftn(f_reconstruct, dim=(-2, -1)))

        return tran_phase