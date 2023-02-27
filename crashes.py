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



pytorch-lightning        1.9.1
torch                    1.13.1
datasets                 2.9.0
transformers             4.25.1
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


def train(model, train_loader, device):
    train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    input_ids, mask, labels = next(iter(train_loader))
    input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)

    outputs = model(input_ids, attention_mask=mask, labels=labels)
    predicted_labels = torch.argmax(outputs["logits"].clone(), 1)
    train_acc.update(predicted_labels, labels)
    train_acc.compute()

    # for attr, default in train_acc._defaults.items():
    #     current_val = getattr(train_acc, attr)
    #     setattr(train_acc, attr, default.to(current_val.device))


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=world_size)

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
        num_workers=4,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=2,
    )

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = DistributedDataParallel(model.to(device), device_ids=[local_rank])

    train(model=model, train_loader=train_loader, device=device)

    torch.distributed.barrier()

    for idx, batch in enumerate(test_loader):
        pass

    torch.distributed.barrier()
    print("completed without errors")
