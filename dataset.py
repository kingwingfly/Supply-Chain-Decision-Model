import torch
from torch.utils.data import DataLoader, Dataset
from conf import DEMAND_RANGE, DEVICE


class DataSet(Dataset):
    def __init__(self, size) -> None:
        super().__init__()
        self.random = torch.randint
        self.size = size

    def __getitem__(self, index) -> None | torch.Tensor:
        if index >= self.size:
            return None
        return self.random(DEMAND_RANGE[0], DEMAND_RANGE[1], (1,)).to(DEVICE)

    def __len__(self):
        return self.size


def new_dataloader(size, batch_size) -> DataLoader:
    return DataLoader(DataSet(size), batch_size=batch_size, shuffle=True)


class SinDataSet(Dataset):
    def __init__(self, size, batch_size) -> None:
        super().__init__()
        start, end = map(
            lambda x: torch.tensor([x], dtype=torch.float, device=DEVICE), DEMAND_RANGE
        )
        mid = (start + end) / 2
        half_range = mid - start
        self.random = lambda x: torch.sin_(torch.tensor(x) * 2 * torch.pi / batch_size).to(
            DEVICE
        ) * half_range + torch.normal(mean=mid, std=half_range / 4).to(DEVICE)
        self.size = size

    def __getitem__(self, index) -> None | torch.Tensor:
        if index >= self.size:
            return None
        return max(self.random(index), torch.zeros(1, device=DEVICE))

    def __len__(self):
        return self.size


def new_sin_dataloader(size, batch_size) -> DataLoader:
    return DataLoader(SinDataSet(size, batch_size), batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    dl = new_sin_dataloader(25, 12)
    a, b = [], []
    import matplotlib.pyplot as plt

    for i in dl:
        a.append(i)
        for j, t in enumerate(i):
            plt.scatter(len(a) * 12 + j, t.item())
    plt.savefig("./sin.png")
    for i in dl:
        b.append(i)
    print(a)
    print(b)
