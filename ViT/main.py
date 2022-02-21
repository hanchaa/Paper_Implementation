import torch
from patch_embedding import PatchEmbedding
from transformer_encoder_block import TransformerEncoderBlock


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    x = torch.randn(8, 3, 224, 224).to(device)
    print(x.shape)

    embedding = PatchEmbedding(3, 16, 768, 224).to(device)
    x = embedding(x)
    print(x.shape)

    encoder = TransformerEncoderBlock(768, 8, 4).to(device)
    x = encoder(x)
    print(x.shape)
