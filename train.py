import torch
from torch import nn
from chain import SupplyChain
import logging
from itertools import chain
from dataset import new_dataloader
import matplotlib.pyplot as plt
import os
from conf import DEVICE

BATCH_SIZE = 12
DATA_SIZE = 120
TOTAL_EPOCH = 20
LR = 1e-3

TARGET = torch.tensor([100], dtype=torch.float).to(DEVICE)
PRE_TRAINED = False


def main():
    global TARGET
    sc = SupplyChain()
    if PRE_TRAINED:
        sc.load_state_dict(torch.load("./pre_models/weight.pth"))
    dt = new_dataloader(DATA_SIZE, BATCH_SIZE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(sc.parameters(), lr=LR)
    max_avg_profit = 0
    best_model = {}
    sc.train()
    for epoch_num in range(TOTAL_EPOCH):
        profits = 0
        for batch_id, demands in enumerate(dt):
            total_profit = sc(demands)
            profits += total_profit.item()
            loss = loss_fn(total_profit / TARGET, torch.ones(1).to(DEVICE))
            plt.scatter(epoch_num, loss.item(), c="black", s=3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc.init()
            print(
                f"epoch num: {epoch_num}\t total_profit: {total_profit.item():.2f}\t loss: {loss.item():.4f}"
            )
        avg_profit = profits / (DATA_SIZE // BATCH_SIZE)
        if avg_profit > max_avg_profit:
            max_avg_profit = avg_profit
            TARGET = torch.tensor(
                max(TARGET.item(), avg_profit * 1.2), dtype=torch.float
            ).to(DEVICE)
            best_model = sc.state_dict()
        print(avg_profit)

    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(best_model, f"./models/weight.pth")
    logging.info("models saved")
    plt.savefig("./losses.png")
    logging.info("losses saved")


if __name__ == '__main__':
    logging.basicConfig(
        filename="chain.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./chain.log")
