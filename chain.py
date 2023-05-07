from __future__ import annotations
import model
import torch
import logging
from model import DEVICE, Model


CONFIGS = [
    {
        'id': 1,
        'initial_stock': 30,
        'selling_price': 3,
        'purchase_price': 2,
        'stock_price': 0.1,
        'compensation': 0.1,
        'handling_fee': 2,
    },
    {
        'id': 2,
        'initial_stock': 30,
        'selling_price': 2,
        'purchase_price': 1.5,
        'stock_price': 0.05,
        'compensation': 0.1,
        'handling_fee': 3,
    },
    {
        'id': 3,
        'initial_stock': 30,
        'selling_price': 1.5,
        'purchase_price': 1.2,
        'stock_price': 0.02,
        'compensation': 0.1,
        'handling_fee': 4,
    },
    {
        'id': 4,
        'initial_stock': 30,
        'selling_price': 1.2,
        'purchase_price': 1,
        'stock_price': 0.01,
        'compensation': 0.1,
        'handling_fee': 3,
    },
]


class OrderForm:
    def __init__(self, ordering_party: Saler | None, order_num: torch.Tensor) -> None:
        self.ordering_party = ordering_party
        self._order_num = order_num
        # logging.debug(self)

    def __str__(self) -> str:
        return f'OrderForm from saler id:{self.ordering_party._id if self.ordering_party else None}\t ordering number is {self.order_num.item()}'

    @property
    def order_num(self):
        return self._order_num


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


class Saler:
    def __init__(
        self,
        id: int,
        initial_stock: int,
        selling_price: int,
        purchase_price: int,
        stock_price: float,
        compensation: float,
        handling_fee: float | int,
        next: Saler | None,
    ) -> None:
        self._id = id
        self._epoch_num = 0
        self._log = {}
        self._stocker = Storehouse(id, initial_stock)
        self._model = model.new_model()
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
        self._next = next

    def init(self):
        self._epoch_num = 0
        self._profit = torch.zeros(1).to(DEVICE)
        self._stocker.init()

    # def __repr__(self) -> str:
    #     return f"{self._id}"

    def __str__(self) -> str:
        return f"id: {self._id}\t stock: {self.stock.item():.0f}\t profit: {self.profit.item():.2f}\t next saler id: {self._next._id if self._next else None}"

    def _deliver(self, order_form: OrderForm):
        order_party = order_form.ordering_party
        afford_num = self._afford_num(order_form.order_num)
        order_num = order_form.order_num if order_party else afford_num
        self.stocker.get(order_num)
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

    def _order(self, order_form: OrderForm) -> torch.Tensor:
        order_num = order_form.order_num
        self._model.train()
        new_order_num: torch.Tensor = self._model(
            torch.tensor(
                [
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
        )
        self._profit -= (
            self._purchase_price * new_order_num + self._handling_fee
            if new_order_num
            else 0 * new_order_num
        )
        if self._next:
            new_order_form = OrderForm(ordering_party=self, order_num=new_order_num)
            logging.debug(
                f"{self._id} order from {self._next._id} number {new_order_num.item()}"
            )
            self._next.epoch(new_order_form)
        else:
            # The final one
            self.stocker.put(new_order_num)
            logging.debug(f"product number {new_order_num.item()}")

    def _afford_num(self, order_num: torch.Tensor):
        return torch.max(
            torch.stack(
                (
                    torch.min(torch.stack((self.stock, order_num)), 0)[0],
                    torch.zeros(1).to(DEVICE),
                )
            ),
            0,
        )[0]

    def _process(self, order_form: OrderForm):
        self._deliver(order_form)
        # print("after deliver", self._id, self._profit.item())
        self._order(order_form)
        # print("after order", self._id, self._profit.item())

    def epoch(self, order_form: OrderForm):
        self._epoch_num += 1
        self._process(order_form)

    @property
    def stocker(self):
        return self._stocker

    @property
    def stock(self):
        return self._stocker.stock

    @property
    def profit(self):
        return self._profit


class SupplyChain:
    def __init__(self, configs) -> None:
        salers = [Saler(next=None, **config) for config in configs]
        for i in range(len(salers)):
            salers[i]._next = salers[i + 1] if i + 1 != len(salers) else None
        self.salers = salers

    def init(self):
        for saler in self.salers:
            saler.init()

    def __str__(self) -> str:
        return '\n'.join(map(str, self.salers))

    def order(self, demand: torch.Tensor):
        order_form = OrderForm(None, demand)
        self.salers[0].epoch(order_form=order_form)

    @property
    def total_profit(self):
        return torch.sum(torch.stack([saler.profit for saler in self.salers]), 0)


def main():
    sc = SupplyChain(configs=CONFIGS)
    predict_demand = torch.tensor([[i] for i in [18, 19, 20, 30, 25, 20, 15]]).to(
        DEVICE
    )
    for demand in predict_demand:
        sc.order(demand=demand)
        logging.info('\n' + str(sc))
    logging.info('total profit:' + str(sc.total_profit.item()))


if __name__ == '__main__':
    logging.basicConfig(
        filename="chain.log", filemode='w', encoding="utf-8", level=logging.DEBUG
    )
    main()
    print("log here: ./chain.log")
