import torch
from torch import nn
from chain import SupplyChain, CONFIGS
from model import DEVICE
import logging
from itertools import chain


def fit(sc: SupplyChain):
    target = torch.tensor([300], dtype=torch.float).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        chain(*[saler._model.parameters() for saler in sc.salers]), lr=5e-3
    )
    # loss = loss_fn(sc.salers[0]._profit / target, torch.ones(1).to(DEVICE))
    loss = loss_fn(sc.total_profit / target, torch.ones(1).to(DEVICE))
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    sc = SupplyChain(configs=CONFIGS)
    predict_demand = torch.tensor(
        [[i] for i in [18, 19, 20, 20, 22, 25, 30, 32, 25, 22, 20, 15]]
    ).to(DEVICE)
    for epoch in range(300):
        for demand in predict_demand:
            sc.order(demand=demand)
            logging.info('\n' + str(sc))
        logging.info('total profit:' + str(sc.total_profit.item()))
        fit(sc)
        sc.init()
        print("epoch:", epoch)


if __name__ == '__main__':
    logging.basicConfig(
        filename="chain.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./chain.log")
