from train_logger import Logger
import wandb

class WandbLogger(Logger):
    def __init__(self, entity, project):
        wandb.init(entity=entity, project=project)

    def log(self, metrics: dict):
        wandb.log(metrics)