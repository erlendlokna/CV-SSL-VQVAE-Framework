import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def training_step(self, batch, batch_idx):
        raise NotImplemented
    
    def validation_step(self, batch, batch_idx):
        raise NotImplemented
    
    def configure_optimizers(self):
        raise NotImplemented


def detach_the_unnecessary(loss_hist: dict):
    """
    apply `.detach()` on Tensors that do not need back-prop computation.
    :return:
    """
    for k in loss_hist.keys():
        if k not in ['loss']:
            try:
                loss_hist[k] = loss_hist[k].detach()
            except AttributeError:
                pass