from chain import SupplyChain, CONFIGS
from conf import DEVICE
import torch


if __name__ == "__main__":
    sc = SupplyChain().to(DEVICE)
    # init
    sc.load_state_dict(torch.load(f"./models/weight.pth"))
    sc.eval()

    # predict
    epoch_num = 1
    while demands := torch.tensor(
        [[eval(input("The customer demand:"))]], dtype=torch.float
    ).to(DEVICE):
        print(f"epoch num: {epoch_num}")
        for saler in sc.salers:
            result = saler.predict_order(demands)
            print(f"{saler._id} should order number {result.item()}")
        sc(demands)
        print(f"total profit:\t {sc.total_profit.item():.2f}")
        for saler in sc.salers:
            print(f"{saler._id}'s profit: {saler.profit.item():.2f}")
        print('\n')
        epoch_num += 1
