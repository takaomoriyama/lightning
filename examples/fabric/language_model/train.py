from torch.utils.data import DataLoader, random_split

from model import LLaMA, LLaMAConfig
import lightning as L

import torch

from pytorch.demos.transformer import WikiText2


def main():
    L.seed_everything(42)

    fabric = L.Fabric()  # accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()

    # Data
    dataset = WikiText2()
    train_data, val_data = get_dataloaders(dataset)

    # Model
    config = LLaMAConfig.from_name("mini")
    config.vocab_size = dataset.vocab_size

    with fabric.device:
        model = LLaMA(config)

    # model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-1)

    model, optimizer = fabric.setup(model, optimizer)
    train_data, val_data = fabric.setup_dataloaders(train_data, val_data)
    train(fabric, model, optimizer, train_data, val_data)


def train(fabric, model, optimizer, train_dataloader, val_dataloader, max_epochs=20):
    for epoch in range(max_epochs):
        train_epoch(fabric, model, optimizer, train_dataloader, epoch)
        val_loss = validate(fabric, model, val_dataloader)
        fabric.print(f"val loss {val_loss.item():.4f}")


def train_epoch(fabric, model, optimizer, train_dataloader, epoch):
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, targets = batch
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 200 == 0:
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - loss {loss.item():.4f}")


@torch.no_grad()
def validate(fabric, model, val_dataloader):
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(len(val_dataloader))
    for k, batch in enumerate(val_dataloader):
        input_ids, targets = batch
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_dataloaders(dataset):
    n = len(dataset)
    train_dataset, val_dataset = random_split(dataset, [n - 2000, 2000])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
