from chain import SupplyChain, CONFIGS, DEVICE
import torch


if __name__ == "__main__":
    sc = SupplyChain(CONFIGS)
    for saler in sc.salers:
        saler._model.load_state_dict(torch.load(f"./models/{saler._id}_weight.pth"))
        saler._model.eval()
    while demand := torch.tensor([eval(input("The customer demand:"))], dtype=torch.float).to(DEVICE):
        for saler in sc.salers:
            result = saler.predict_order(demand)
            print(f"{saler._id} should order number {result.item()}")