import torch
from patch_embedding import PatchEmbedding
from multi_head_attention import MultiHeadAttention


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    x = torch.randn(8, 3, 224, 224).to(device)
    print(x.shape)

    patch_embedding = PatchEmbedding(3, 16, 768, 224).to(device)

    x = patch_embedding(x)
    print(x.shape)

    mha = MultiHeadAttention(768, 8).to(device)

    x = mha(x)
    print(x.shape)
