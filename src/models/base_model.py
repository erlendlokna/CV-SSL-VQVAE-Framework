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
