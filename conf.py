import platform
import torch


print(platform.platform())
print(torch.__version__)

DEMAND_RANGE = (10, 40)


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


DEVICE, BACKEND = (
    ("cuda", "inductor" if "Win" not in platform.platform() else None)
    if torch.cuda.is_available()
    else ("mps", "aot_eager")
    if torch.backends.mps.is_available()
    else ("cpu", "inductor")
)
print(f"Using {DEVICE} device")
