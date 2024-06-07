import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as ptl

wandb.login()

config = {
    "name": "Experiment-1",
    "dataroot": "./datasets/temp_value",
    "batchSize": 1,
    "max_epoch": 100,
    "optimizer": "Adam",
    "lr": 1e-3,
    "gpu_ids": "-1",
    "norm": "instance",
    "dropout": False,
    "num_patients": 10,
    "num_modalities": 4,
    "latent_dim": 256,
    "num_spatial_freq": 9,
    "num_anatomy_blocks": 2,
    "num_modality_blocks": 1
}

run = wandb.init(
    project="Disentangled INF for MRI Translation",
    config=config
)

wandb_logger = WandbLogger(project="Disentangled INF for MRI Translation",
                           name="Experiment-1",
                           log_model="all",
                           save_dir="wandb_logs")

checkpoint_callback = ptl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")

# Use to add config to wandb
"""wandb_logger.experiment.config.update({
   "key": value 
})"""

trainer = ptl.Trainer(logger=wandb_logger, callback=[checkpoint_callback])

