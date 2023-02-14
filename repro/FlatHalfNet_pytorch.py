import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from Components import DataComponents
from Components import ModuleComponents
import torch.utils.tensorboard
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('lightning_logs', name='FlatHalfNet_run')


device = "cuda" if torch.cuda.is_available() else "cpu"


class HalfNet(nn.Module):

    def __init__(self):
        super(HalfNet, self).__init__()
        channel_base = 8
        self.inc = ModuleComponents.DoubleConv3D(1, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.down1 = ModuleComponents.Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge1 = ModuleComponents.Merge3D(2)
        self.down2 = ModuleComponents.Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge2 = ModuleComponents.Merge3D(4)
        self.down3 = ModuleComponents.Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge3 = ModuleComponents.Merge3D(8)
        self.down4 = ModuleComponents.Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge4 = ModuleComponents.Merge3D(16)
        self.down5 = ModuleComponents.Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge5 = ModuleComponents.Merge3D(32)
        self.z_work = ModuleComponents.DoubleConv3D(channel_base, channel_base, ghost=False, kernel_size=(3, 1, 1))
        self.outc = ModuleComponents.OutConv3D(channel_base, 2)

    def forward(self, x):
        x = self.inc(x)
        x2 = self.down1(x)
        x = self.merge1(x, x2)
        x2 = self.down2(x2)
        x = self.merge2(x, x2)
        x2 = self.down3(x2)
        x = self.merge3(x, x2)
        x2 = self.down4(x2)
        x = self.merge4(x, x2)
        x2 = self.down5(x2)
        x = self.merge5(x, x2)
        x = self.z_work(x)
        x = self.outc(x)
        return x


class HalfNetPL(pl.LightningModule):

    def __init__(self, learning_rate=0.001):
        super(HalfNetPL, self).__init__()
        self.model = HalfNet()
        self.learning_rate = learning_rate

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(input=y_hat, target=y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    train_dataset = DataComponents.Train_Dataset('datasets/train/img',
                                                 'datasets/train/lab',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1, logger=False,
                         accelerator="cpu", devices=1, enable_checkpointing=False,
                         precision=16)
    model = HalfNetPL()
    trainer.fit(model, train_dataloaders=train_loader)
