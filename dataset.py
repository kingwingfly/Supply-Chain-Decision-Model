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


if __name__ == "__main__":
    dl = new_dataloader(13, 12)
    for i in dl:
        print(i)
