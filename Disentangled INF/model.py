import numpy as np
import torch.nn as nn
import pytorch_lightning as ptl

from network import *


class MRITranslationINF(ptl.LightningModule):
    def __init__(self, config):
        super(MRITranslationINF, self).__init__()
        self.save_hyperparameters(config)

        self.model = DisentangledINF(
            num_anatomy_blocks=config["num_anatomy_blocks"],
            num_modality_blocks=config["num_modality_blocks"],
            encoding_dim=config["latent_dim"],
            num_spatial_freq=config["num_spatial_freq"],
            latent_dim=config["latent_dim"]
        )

        # Training counters
        self.step, self.cnt_train_step, self.cnt_test_step = self.current_epoch, self.global_step, self.global_step

    def training_step(self, batch, batch_idx):
        pass
