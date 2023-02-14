import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from Components import DataComponents
from Components import ModuleComponents


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


def dummy_train():
    train_dataset = DataComponents.Train_Dataset('datasets/train/img', 'datasets/train/lab',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)

    model = HalfNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in enumerate(train_loader):
            x, y = batch
            y_hat = model(x)
            loss = F.cross_entropy(input=y_hat, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    dummy_train()
