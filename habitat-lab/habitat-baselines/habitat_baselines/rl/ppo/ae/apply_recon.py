import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae import AE
import numpy as np

class apply_ae():
    def __init__(self, device, adapt_encoder, apply_ablation=False, ablation_block=None):
        self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
        self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:512"
        self.model = AE(input_res=256, dec_block_str=self.dec_block_config_str, dec_channel_str=self.dec_channel_config_str)
        
        self.state_dict = torch.load('/mnt/iusers01/fatpou01/compsci01/n70579mp/habitat-lab/data/autoencoder/ae_3rd_encoder.pt')
        self.model.load_state_dict(self.state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.adapt_encoder = adapt_encoder
        self.apply_ablation = apply_ablation
        self.ablation_block = ablation_block
        # DUA
        self.mom_pre = 0.1
        self.decay_factor = 0.94
        self.min_mom = 0.005

        # count for printing the first step (debugging purpose)
        self.count = 0
    def recon(self, observation):
        self.model.eval()
        # Use DUA to adapt encoder
        if self.adapt_encoder:
            if not(self.apply_ablation):
                self.model, self.mom_pre, self.decay_factor, self.min_mom = self._adapt(self.model, 
                                                                                            self.mom_pre, 
                                                                                            self.decay_factor, 
                                                                                            self.min_mom)
            else: # Ablation Study
                self.model, self.mom_pre, self.decay_factor, self.min_mom = self._adapt_ablation(self.model, 
                                                                                            self.mom_pre, 
                                                                                            self.decay_factor, 
                                                                                            self.min_mom)
        observation = observation / 255.0
        x = torch.from_numpy(observation).permute(2, 0, 1).float().unsqueeze(dim=0).to(self.device)
        decoder_out = self.model(x)
        decoder_out = decoder_out.squeeze().permute(1, 2, 0)
        decoder_out = decoder_out.cpu().detach().numpy()
        decoder_out = (decoder_out*255).clip(0, 255).astype(np.uint8)
        self.model.eval()
        return decoder_out

    def _adapt(self, 
                model,
                mom_pre,
                decay_factor,
                min_mom):
        encoder = model.enc
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        for m in encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return model, mom_pre, decay_factor, min_mom

    def _adapt_ablation(self, 
                model,
                mom_pre,
                decay_factor,
                min_mom):
        
        encoder = model.enc
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        encoder_block = getattr(encoder.backbone, f"layer{self.ablation_block}")
        if self.count == 0:
            print(f"Ablation: allow only block {self.ablation_block} to update norm", encoder_block)
            self.count += 1
        for m in encoder_block.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return model, mom_pre, decay_factor, min_mom

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))
    print(f"device {device}")
    model = apply_ae(device)
    print("DONE")
