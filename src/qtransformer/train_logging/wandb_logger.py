from qtransformer.train_logging.traín_logger import Logger
import wandb

def flatten_config_dict(config, prefix=None):
    if prefix:
        prefix += "_"
    else:
        prefix = ""
    result = dict()
    for k, v in config.items():
        if isinstance(v, dict):
            sub = flatten_config_dict(v, prefix=prefix + k)
            result.update(sub)
        else:
            result[prefix + k] = v

    return result


class WandbLogger(Logger):
    def __init__(self, entity, project, config):
        wandb.init(entity=entity, project=project, config=flatten_config_dict(config))
        self.curr = dict()

    def log_metrics(self, metrics: dict):
        self.curr.update(metrics)

    def write_step(self):
        wandb.log(self.curr)
        self.curr = dict()