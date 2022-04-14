from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embedding_size: int = 768, num_classes: int = 1000):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        out = x[:, 0]
        out = self.layer_norm(out)
        out = self.linear(out)

        return out
