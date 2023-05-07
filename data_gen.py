import torch
from model import DEVICE


class DataSet:
    def __init__(self, size) -> None:
        self.random = torch.randint
        self.size = size

    def __iter__(self):
        for _ in range(self.size):
            yield self.random(10, 40, (12, 1)).to(DEVICE)


if __name__ == "__main__":
    ds = DataSet()
    for i in ds:
        print(i)
