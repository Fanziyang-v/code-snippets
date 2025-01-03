from torch import nn
from torchvision import models


def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    model = models.resnet18()
    print(f"Total number of parameters: {count_parameters(model)}")


if __name__ == "__main__":
    main()
