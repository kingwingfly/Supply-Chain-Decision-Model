import torch
from torch import nn
from chain import SupplyChain, CONFIGS
from model import DEVICE
import logging
from itertools import chain
from data_gen import DataSet
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 40
TOTAL_EPOCH = 80
LR = 1e-3

TARGET = torch.tensor([600], dtype=torch.float).to(DEVICE)
LOSSES = []


def fit(sc: SupplyChain):
    global TARGET, LOSSES
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        chain(*[saler._model.parameters() for saler in sc.salers]), lr=LR
    )
    TARGET = torch.tensor(
        [max(TARGET.item(), sc.total_profit.item() * 1.2)], dtype=torch.float
    ).to(DEVICE)
    loss = loss_fn(sc.salers[0]._profit / TARGET, torch.ones(1).to(DEVICE))
    loss = loss_fn(sc.total_profit / TARGET, torch.ones(1).to(DEVICE))
    # print(loss.item())
    LOSSES.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    sc = SupplyChain(configs=CONFIGS)
    for saler in sc.salers:
        saler._model.load_state_dict(torch.load(f"./pre_models/{saler._id}_weight.pth"))
    dt = DataSet(BATCH_SIZE)
    max_avg_profit = 0
    max_profit = 0
    best_models = {}
    for epoch in range(TOTAL_EPOCH):
        profits = 0
        for predict_demand in dt:
            for demand in predict_demand:
                sc.order(demand=demand)
                logging.info('\n' + str(sc))
            logging.info('total profit:' + str(sc.total_profit.item()))
            profits += sc.total_profit.item()
            max_profit = max(max_profit, sc.total_profit.item())
            fit(sc)
            sc.init()
        print("epoch:", epoch, "finished")
        avg_profit = profits / BATCH_SIZE
        print("current_avg_profit:", avg_profit)
        if avg_profit > max_avg_profit:
            max_avg_profit = avg_profit
            best_models = dict(
                [(saler._id, saler._model.state_dict()) for saler in sc.salers]
            )
        print("max_avg_profit:", max_avg_profit)
        print("max_profit:", max_profit)
        print("TARGET:", TARGET.item())

    if not os.path.exists("./models"):
        os.makedirs("./models")
    for id_, state_dict in best_models.items():
        torch.save(state_dict, f"./models/{id_}_weight.pth")
    logging.info("models saved")
    for x in range(TOTAL_EPOCH):
        ys = LOSSES[BATCH_SIZE * x : BATCH_SIZE * (x + 1)]
        for y in ys:
            plt.scatter(x, y, c="black", s=3)
    plt.savefig("./losses.png")
    logging.info("losses saved")


if __name__ == '__main__':
    logging.basicConfig(
        filename="chain.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./chain.log")
