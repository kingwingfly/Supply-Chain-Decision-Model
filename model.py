import torch
from torch import nn

print(torch.__version__)

DEVICE, BACKEND = (
    ("cuda", "inductor")
    if torch.cuda.is_available()
    else ("mps", "aot_eager")
    if torch.backends.mps.is_available()
    else ("cpu", "inductor")
)
print(f"Using {DEVICE} device")


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        logits = self.linear_relu_stack(x)
        return logits * 30


def new_model() -> Model:
    model = Model().to(device=DEVICE)
    model = torch.compile(model, backend=BACKEND)
    return model
