
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class EfficientClassifier(nn.Module):
    def __init__(self, num_classes=2, model_version="efficientnet-b0"):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_version, num_classes=num_classes)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x, epoch=None):
        return self.out_act(self.model(x))


if __name__ == "__main__":
    model = EfficientClassifier()