from chain import SupplyChain, CONFIGS, DEVICE
import torch


if __name__ == "__main__":
    sc = SupplyChain(CONFIGS)
    # init
    for saler in sc.salers:
        saler._model.load_state_dict(torch.load(f"./models/{saler._id}_weight.pth"))
        saler._model.eval

    # predict
    epoch_num = 1
    while demand := torch.tensor([eval(input("The customer demand:"))], dtype=torch.float).to(DEVICE):
        print(f"\nepoch num: {epoch_num}")
        for saler in sc.salers:
            result = saler.predict_order(demand)
            print(f"{saler._id} should order number {result.item()}")
        sc.order(demand=demand)
        print(f"total profit:\t {sc.total_profit.item():.2f}")
        for saler in sc.salers:
            print(f"{saler._id}'s profit: {saler.profit.item():.2f}")
        epoch_num += 1