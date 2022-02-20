import torch
from patch_embedding import PatchEmbedding


x = torch.randn(8, 3, 224, 224)
print(x.shape)

patch_embedding = PatchEmbedding(3, 16, 768, 224)

x = patch_embedding(x)
print(x.shape)
