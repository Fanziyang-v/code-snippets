"""
This code snippet shows an simple example on training model 
using multiple GPUs by DDP(DistributedDataParallel).

DDP helps to keep the model replicas the same by gradients sychronization.

The process to work with DDP are as follows.
1. Initialize the process group.
2. Create data loader with DistributedSampler(dataset).
3. Wrap the model with DDP.
4. Train the model.
5. Destroy the process group.
"""

import os
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# DDP-related packages
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


class SimpleModel(nn.Module):
    """Simple CNN Model."""

    def __init__(self, num_channels: int = 1, num_classes: int = 10) -> None:
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _ddp_setup(rank: int, world_size: int, gpu_id: int) -> None:
    """DDP setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # torch.cuda.set_device(gpu_id)
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def _build_dataset() -> Dataset:
    # MNIST dataset
    return datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [transforms.RandomCrop(28, padding=4), transforms.ToTensor()]
        ),
    )


def _build_data_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,  # DistributedSampler will shuffle the datasets.
        pin_memory=True,
        num_workers=4,
        sampler=DistributedSampler(
            dataset
        ),  # Restrict each process loads a subset of the dataset.
    )


def main(rank: int, gpu_ids: list[int]) -> None:
    gpu_id = gpu_ids[rank]
    world_size = len(gpu_ids)
    _ddp_setup(rank, world_size, gpu_id)
    dataset = _build_dataset()
    data_loader = _build_data_loader(dataset)
    model = SimpleModel().to(gpu_id)
    criterion = nn.CrossEntropyLoss().to(gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = DDP(model, device_ids=[gpu_id])
    # start training
    for i in range(20):
        total_loss = total_acc = 0
        for images, labels in data_loader:
            images: Tensor = images.to(gpu_id)
            labels: Tensor = labels.to(gpu_id)

            # Forward pass
            outputs = model(images)
            pred_labels = torch.argmax(outputs, dim=1)
            total_acc += (pred_labels == labels).float().mean()
            loss: Tensor = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"""[GPU{gpu_id}]=> \
Epoch: [{i + 1}]/[{20}], \
loss: {total_loss / len(data_loader): .4f}, \
acc:{total_acc / len(data_loader) * 100 : .2f} %"""
        )
    # only one model replica needs to save model checkpoints.
    if rank == 0:
        torch.save(model.module.state_dict(), "simple_model.pth")
    destroy_process_group()


if __name__ == "__main__":
    gpu_ids = [0, 1, 2, 3]  # four GPU
    num_procs = len(gpu_ids)
    # create num_procs processes such that each process uses a single GPU.
    mp.spawn(main, args=(gpu_ids,), nprocs=num_procs)
