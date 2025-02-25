import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

class Model(pl.LightningModule):
    def __init__(self, vocab_size, model):
        super().__init__()
        self.model = model(vocab_size)

    def train(self, mode = True):
        return super().train(mode)
    
    def forward(self, index, targets=None):
        return self.model(index, targets)        
    
    def training_step(self, batch, batch_idx):  
        x, y = batch
        logits, loss = self(x, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        return loss
    
    def configure_optimizers(self): 
        return optimizer
