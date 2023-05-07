import torch
from torch import nn
import platform

print(platform.platform())
print(torch.__version__)

DEVICE, BACKEND = (
    ("cuda", "inductor" if "Win" not in platform.platform() else None)
    if torch.cuda.is_available()
    else ("mps", "aot_eager")
    if torch.backends.mps.is_available()
    else ("cpu", "inductor")
)
print(f"Using {DEVICE} device")


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.lstm = nn.LSTM(7, 8)
        self.linear = nn.Linear(8,1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.view(1,-1))
        out = self.linear(out).view(-1)
        out = self.relu(out)
        return out * 40


def new_model() -> Model:
    model = Model().to(device=DEVICE)
    if BACKEND:
        model = torch.compile(model, backend=BACKEND)
    return model
