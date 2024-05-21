import numpy as np
import torch.nn as nn
import pytorch_lightning as ptl

from network import *


class MRITranslationINF(ptl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("INF_MRITranslation")
        parser.add_argument('--num_anatomy_blocks', type=int, default=2, help='number of anatomy blocks')
        parser.add_argument('--num_modality_blocks', type=int, default=2, help='number of modality blocks')
        parser.add_argument('--encoding_dim', type=int, default=256, help='encoding dimension')
        parser.add_argument('--num_spatial_freq', type=int, default=10, help='number of spatial frequencies')
        parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')
        return parent_parser

    def __init__(self, **kwargs):
        super(MRITranslationINF, self).__init__()
        self.save_hyperparameters()
        self.model = DisentangledINF(**self.hparams)

        # Training counters
        self.step, self.cnt_train_step, self.cnt_test_step = self.current_epoch, self.global_step, self.global_step

    def training_step(self, batch, batch_idx):
        pass
