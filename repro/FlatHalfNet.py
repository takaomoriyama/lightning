# 基于: https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full#F2
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
        #self.automatic_optimization = False

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        #loss = self._step(batch, batch_idx)
        #self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': self._step(batch, batch_idx)}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return {'val_loss': loss}

    def configure_optimizers(self):
        # Adam默认学习率就是0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                               threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0.00025, verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch
        return torch.argmax(self(x), dim=1)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.val_loss = avg_loss
        epoch_dictionary={
            'val_loss': avg_loss,}
        #return epoch_dictionary
    
    def training_epoch_end(self, outputs):
        #if(self.current_epoch==0):
            #sampleImg = torch.rand((1,1,5,128,128))
            #self.logger.experiment.add_graph(HalfNet(), sampleImg)
            #print("HeySeeItShouldWork")
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        epoch_dictionary={
            'loss': avg_loss,}
        #return epoch_dictionary

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss}
        
    def _step(self, batch, batch_idx):
        #opt = self.optimizers()
        #opt.zero_grad()
        x, y = batch
        y_hat = self.forward(x)
        #weight = torch.tensor(data=(1, 1), device=self.device)
        #weight = weight.to(torch.float)
        loss = F.cross_entropy(input=y_hat, target=y)
        #self.manual_backward(loss)
        #self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        #opt.step()
        #sch = self.lr_schedulers()
        #sch.step()
        return loss


if __name__ == "__main__":
    #载入数据
    train_dataset = DataComponents.Train_Dataset('datasets/train/img',
                                                 'datasets/train/lab',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    val_dataset = DataComponents.Val_Dataset('datasets/val/img',
                                             'datasets/val/lab',)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=1, persistent_workers=True)
    # Setting up training parameters
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1, logger=logger,
                         accelerator="cpu", enable_checkpointing=False,
                         precision=16, auto_lr_find=True, gradient_clip_val=0.5,)
    model = HalfNetPL()
    #model = torch.load('model_flathalfnetv0.pth')
    #trainer.tune(model, train_dataloaders=train_loader)
    # Train the model
    trainer.fit(model,
                val_dataloaders=val_loader,
                train_dataloaders=train_loader)
    trainer.test(model,
                 dataloaders=test_loader)
    # 保存模型
    torch.save(model, 'model_flathalfnetv0.pth')
    # 输出预测
    predict_dataset = DataComponents.Predict_Dataset('datasets/predict', vol=True)
    predict_loader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=1, num_workers=0)
    # predictions = 一个list，包含了维度是(b,深度,高度,宽度)的输出张量(没有C！)
    original_volume = predict_dataset.__getoriginalvol__()
    predictions = trainer.predict(model, predict_loader)
    DataComponents.predictions_to_final_img(predictions, direc="datasets/result", original_volume=original_volume, vol_out=True)