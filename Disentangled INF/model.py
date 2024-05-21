import numpy as np
import torch.nn as nn
import pytorch_lightning as ptl

from network import *


class MRITranslationINF(ptl.LightningModule):
    def __init__(self, **kwargs):
        super(MRITranslationINF, self).__init__()
        self.save_hyperparameters()
        self.latent_dim = 256
        self.num_anatomy_blocks = 2
        self.num_modality_blocks = 1
        # 2^9 = 512; Nyquist sampling theorem tells us this will capture all info for 256^3 volume
        self.num_spatial_freq = 9
        self.model = DisentangledINF(**self.hparams)
        self.anatomy_latent_codes = nn.Embedding(self.num_patients, self.latent_dim)
        self.modality_latent_codes = nn.Embedding(self.num_modalities, self.latent_dim)

        # Training counters
        self.step, self.cnt_train_step, self.cnt_test_step = self.current_epoch, self.global_step, self.global_step

    def training_step(self, batch, batch_idx):
        pass
