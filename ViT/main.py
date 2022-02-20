import torch
from patch_embedding import PatchEmbedding


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    x = torch.randn(8, 3, 224, 224).to(device)
    print(x.shape)

    patch_embedding = PatchEmbedding(3, 16, 768, 224).to(device)

    x = patch_embedding(x)
    print(x.shape)
