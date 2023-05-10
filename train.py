import torch
from torch import nn
from chain import SupplyChain
import logging
from dataset import new_dataloader
import matplotlib.pyplot as plt
import os
from conf import DEVICE
import math

BATCH_SIZE = 8
DATA_SIZE = 120
TOTAL_EPOCH = 10
LR = 1e-3

PRE_TRAINED = True


def main():
    sc = SupplyChain().to(DEVICE)
    if PRE_TRAINED:
        sc.load_state_dict(torch.load("./models/weight.pth"))
    train_dl = new_dataloader(DATA_SIZE, BATCH_SIZE)
    valid_dl = new_dataloader(DATA_SIZE, BATCH_SIZE)
    loss_fn = lambda x: -x
    optimizer = torch.optim.Adam(sc.parameters(), lr=LR)
    max_avg_profit = 0
    best_model = {}
    for epoch_num in range(1, TOTAL_EPOCH + 1):
        profits = 0
        sc.train()
        for batch_id, demands in enumerate(train_dl):
            total_profit = sc(demands)
            profits += total_profit.item()
            loss: torch.Tensor = loss_fn(total_profit)
            # loss = loss_fn(sc.salers[0]._profit)
            plt.scatter(epoch_num, loss.item(), c="black", s=3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc.init()
            logging.info(
                f"epoch num: {epoch_num}\t batch_id: {batch_id}, total_profit: {total_profit.item():.2f}\t loss: {loss.item():.4f}"
            )
            save_config()

        avg_profit = profits / (math.ceil(DATA_SIZE / BATCH_SIZE))
        logging.info(f"epoch num: {epoch_num}\t avg_profit: {avg_profit:.2f}")
        print(f"epoch num: {epoch_num}\t avg_profit: {avg_profit:.2f}")
        if avg_profit > max_avg_profit:
            max_avg_profit = avg_profit
            best_model = sc.state_dict()
        logging.info(f"epoch num: {epoch_num}\t max_avg_profit: {max_avg_profit:.2f}")
        print(f"epoch_num: {epoch_num}\t max_avg_profit: {max_avg_profit:.2f}")

        # sc.eval()
        # profits = 0
        # for _, demands in enumerate(valid_dl):
        #     total_profit = sc(demands)
        #     profits += total_profit.item()
        #     sc.init()
        # print(f"valid avg_profit: {profits / math.ceil(DATA_SIZE / BATCH_SIZE):.2f}")

    if 'y' == input("save model?[y/N]\t").strip().lower():
        save_model(best_model)


def save_model(best_model):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(best_model, "./models/weight.pth")
    logging.info("models saved")


def save_config():
    plt.savefig("./static/losses.png")


if __name__ == '__main__':
    logging.basicConfig(
        filename="train.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./train.log")
