"""
Run this script with:

    torchrun --nproc_per_node 2 --standalone crashes.py

to reproduce the error:

Traceback (most recent call last):
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1120, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/queues.py", line 122, in get
    return _ForkingPickler.loads(res)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/multiprocessing/reductions.py", line 305, in rebuild_storage_fd
    fd = df.detach()
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/resource_sharer.py", line 86, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/connection.py", line 513, in Client
    answer_challenge(c, authkey)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/connection.py", line 757, in answer_challenge
    message = connection.recv_bytes(256)         # reject large message
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/connection.py", line 221, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/connection.py", line 419, in _recv_bytes
    buf = self._recv(4)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/multiprocessing/connection.py", line 384, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/adrian/repositories/lightning/crashes.py", line 230, in <module>
    for idx, batch in enumerate(test_loader):
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1316, in _next_data
    idx, data = self._get_data()
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1282, in _get_data
    success, data = self._try_get_data()
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1120, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/adrian/anaconda3/envs/lightning/lib/python3.9/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2339965) is killed by signal: Aborted.

"""

from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel

import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


import os

from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


def train(num_epochs, model, optimizer, train_loader, device):

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            model.train()

            if batch_idx > 3:
                break

            for s in range(3):
            # for s in ["input_ids", "attention_mask", "label"]:
                batch[s] = batch[s].to(device)
                print(s, batch[s].shape)

            # outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
            outputs = model(batch[0], attention_mask=batch[1], labels=batch[2])


            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(outputs["logits"].clone(), 1)
                # train_acc.update(predicted_labels, batch["label"])
                train_acc.update(predicted_labels, batch[2].clone())

        print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}%")
        train_acc._update_count = 0
        train_acc._forward_cache = None
        train_acc._computed = None

        print(train_acc._defaults)
        for attr, default in train_acc._defaults.items():
            current_val = getattr(train_acc, attr)
            if isinstance(default, torch.Tensor):
                setattr(train_acc, attr, default.detach().clone().to(current_val.device))
            # else:
            #     setattr(train_acc, attr, [])

        # reset internal states
        train_acc._cache = None
        train_acc._is_synced = False
        torch.distributed.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=2)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    # del imdb_dataset
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    torch.distributed.barrier()

    train_dataset = torch.utils.data.TensorDataset(
        torch.zeros(100, 512, dtype=torch.int64),
        torch.zeros(100, 512, dtype=torch.int64),
        torch.zeros(100, dtype=torch.int64),
    )
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
    )

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    train(
        num_epochs=1,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
    )

    torch.distributed.barrier()

    # test_dataset = torch.utils.data.TensorDataset(
    #     torch.zeros(5000, 512, dtype=torch.int64),
    #     torch.zeros(5000, 512, dtype=torch.int64),
    #     torch.zeros(5000, dtype=torch.int64),
    # )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=2,
    )

    """
    input_ids torch.int64 torch.Size([12, 512])
    attention_mask torch.int64 torch.Size([12, 512])
    label torch.int64 torch.Size([12])
    """
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(test_loader):
            # torch.save(batch, "batch.pt")
            for s in ["input_ids", "attention_mask", "label"]:
            # print(len(batch))
            # for s in range(3):
                batch[s] = batch[s].to(device)
                print(s, batch[s].dtype, batch[s].shape)

            # outputs = model(batch[0], attention_mask=batch[1], labels=batch[2])
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])

            predicted_labels = torch.argmax(outputs["logits"], 1)
            print("rank", local_rank, "update test_acc", idx)

    torch.distributed.barrier()
