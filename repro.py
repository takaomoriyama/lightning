import os
import torch
import lightning.pytorch as pl

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

class JitCheckpoint(pl.callbacks.Checkpoint):
    def __init__(self):
        self.lastLoss = 100000

    def on_validation_end(self, trainer, pl_module):
        ckpdir      = trainer.log_dir + "/checkpoints/"
        if trainer.is_global_zero:
            print("TorchScript and ONNX checkpoint")
            assert "valid_loss" in trainer.callback_metrics, "monitor not in callback_metrics"
            loss = trainer.callback_metrics["valid_loss"].item()
            if loss < self.lastLoss:
                print("\nSaving checkpoint last loss {} new loss {}".format(self.lastLoss, loss))
                self.lastLoss = loss
                
                filePath    = ckpdir + "epoch={}_loss={:.4f}".format(trainer.current_epoch, loss)
                jitPath     = filePath + '.pt'
                onnxPath    = filePath + '.onnx'
                os.makedirs(ckpdir, exist_ok=True)

                input = torch.randn(10, 32)
                pl_module.to_torchscript(file_path=jitPath, method='trace', example_inputs=[input])
                pl_module.to_onnx(file_path=onnxPath, input_sample=input)
        trainer.strategy.barrier()


def run():
    train_data  = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data    = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data   = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        callbacks=[JitCheckpoint()],
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    run()