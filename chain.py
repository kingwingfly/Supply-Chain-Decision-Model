from __future__ import annotations
import torch
import logging
from torch import nn
from conf import DEVICE, CONFIGS


class Storehouse:
    def __init__(
        self,
        id: int,
        initial_stock: int,
    ) -> None:
        self._id = id
        self._initial_stock = initial_stock
        self._stock = torch.tensor([initial_stock], dtype=torch.float).to(DEVICE)

    def init(self):
        self._stock = torch.tensor([self._initial_stock], dtype=torch.float).to(DEVICE)

    @property
    def stock(self):
        return self._stock

    @stock.setter
    def stock(self, new: torch.Tensor):
        self._stock = new

    def put(self, num: torch.Tensor):
        self.stock += num

    def get(self, num: torch.Tensor):
        self.stock -= num


class Saler(nn.Module):
    def __init__(
        self,
        id: int,
        initial_stock: int,
        selling_price: int,
        purchase_price: int,
        stock_price: float,
        compensation: float,
        handling_fee: float | int,
    ) -> None:
        super().__init__()
        self._id = id
        self._epoch_num = 0
        self._stocker = Storehouse(id, initial_stock)

        self.lstm = nn.LSTM(8, 8)
        self.fc = nn.Sequential(
            nn.Linear(8, 1),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

        self._profit = torch.zeros(1).to(DEVICE)  # 利润
        self._selling_price = torch.tensor([selling_price], dtype=torch.float).to(
            DEVICE
        )
        self._purchase_price = torch.tensor([purchase_price], dtype=torch.float).to(
            DEVICE
        )
        self._stock_price = torch.tensor([stock_price], dtype=torch.float).to(DEVICE)
        self._compensation = torch.tensor([compensation], dtype=torch.float).to(DEVICE)
        self._handling_fee = torch.tensor([handling_fee], dtype=torch.float).to(DEVICE)
        # For salers are connected to each other,
        # if do not wrap the next and prev saler with list,
        # the parameters() or state_dict() method will lead infinite recursion
        self._p_n: list[Saler] = []

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=50)
            if module.bias is not None:
                module.bias.data.zero_()

    def init(self):
        self._epoch_num = 0
        self._profit = torch.zeros(1).to(DEVICE)
        self._stocker.init()

    # def __repr__(self) -> str:
    #     return f"{self._id}"

    def __str__(self) -> str:
        return f"id: {self._id}\t stock: {self.stock.item():.0f}\t profit: {self.profit.item():.2f}\t next saler id: {self._next._id if self._next else None} prev saler id: {self._prev._id if self._prev else None}"

    def _deliver(self, demand: torch.Tensor):
        order_party = self._prev
        afford_num = self._afford_num(demand)
        order_num = demand if order_party else afford_num
        self.stocker.get(order_num)
        # because stock increasement happens **after** paying although stock decreasement happens while being paid, so cannot use setter's side effect to modify the profit here
        self._profit += self._selling_price * order_num
        self._profit -= (
            self._stock_price
            * torch.max(torch.stack((self.stock, torch.zeros(1).to(DEVICE))), 0)[0]
        )
        # the first do not need compensation
        if order_party:
            compensation = (
                -self._compensation
                * torch.min(torch.stack((self.stock, torch.zeros(1).to(DEVICE))), 0)[0]
            )
            self._profit -= compensation
            order_party._profit += compensation
            order_party.stocker.put(afford_num)
            logging.debug(
                f"{self._id} deliver to {order_party._id} number {afford_num.item()}"
            )
        else:
            logging.debug(f"{self._id} deliver to customer number {afford_num.item()}")

    def _order(self, demand: torch.Tensor) -> torch.Tensor:
        order_num = demand
        new_order_num: torch.Tensor = self.predict_order(order_num=order_num)
        self._profit -= (
            self._purchase_price * new_order_num + self._handling_fee
            if new_order_num  # tensor with only one element can be turned into a boolean implicitly
            else 0 * new_order_num
        )
        if self._next:
            logging.debug(
                f"{self._id} order from {self._next._id} number {new_order_num.item()}"
            )
            # self._next.forward(new_order_num)
            return new_order_num
        else:
            # The final one
            self.stocker.put(new_order_num)
            logging.debug(f"product number {new_order_num.item()}")

    def _afford_num(self, order_num: torch.Tensor) -> torch.Tensor:
        return torch.max(
            torch.stack(
                (
                    torch.min(torch.stack((self.stock, order_num)), 0)[0],
                    torch.zeros(1).to(DEVICE),
                )
            ),
            0,
        )[0]

    def _process(self, demand: torch.Tensor):
        self._deliver(demand)
        # print("after deliver", self._id, self._profit.item())
        return self._order(demand)
        # print("after order", self._id, self._profit.item())

    def forward(self, demand: torch.Tensor):
        self._epoch_num += 1
        return self._process(demand)

    def predict_order(self, order_num: torch.Tensor):
        x = torch.tensor(
            [
                self._id,
                self.stock,
                order_num,
                self._selling_price,
                self._purchase_price,
                self._stock_price,
                self._compensation,
                self._handling_fee,
            ],
            dtype=torch.float,
        ).to(DEVICE)
        out, _ = self.lstm(x.view(1, -1))
        out = self.fc(out).view(-1)
        return out

    @property
    def stocker(self):
        return self._stocker

    @property
    def stock(self):
        return self._stocker.stock

    @property
    def profit(self):
        return self._profit

    @property
    def _next(self):
        return self._p_n[0]

    @property
    def _prev(self):
        return self._p_n[1]


class SupplyChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        salers = [Saler(**config).to(DEVICE) for config in CONFIGS]
        for i in range(len(salers)):
            salers[i]._p_n.append(salers[i + 1] if i + 1 != len(salers) else None)
            salers[i]._p_n.append(salers[i - 1] if i != 0 else None)
        self.salers = salers
        self.head = nn.Sequential(*self.salers)

    def init(self):
        for saler in self.salers:
            saler.init()

    def __str__(self) -> str:
        return '\n'.join(map(str, self.salers))

    def forward(self, demands: torch.Tensor) -> torch.Tensor:
        for demand in demands:
            self.head(demand)
            logging.debug('\n' + str(self))
            logging.debug(f"total profit: {self.total_profit.item():.2f}")
        return self.total_profit

    @property
    def total_profit(self):
        return torch.sum(torch.stack([saler.profit for saler in self.salers]), 0)


def main():
    sc = SupplyChain().to(DEVICE)
    demands = torch.tensor(
        [[i] for i in [18, 19, 20, 30, 25, 18, 19, 20, 30, 25, 20, 15]],
    ).to(DEVICE)
    sc(demands=demands)
    logging.info('\n' + str(sc))
    logging.info('total profit:' + str(sc.total_profit.item()))


if __name__ == '__main__':
    logging.basicConfig(
        filename="chain.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./chain.log")
