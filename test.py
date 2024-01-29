import torch
import torch.nn.functional as F

# Assume `scores` are the attention scores, and `mask` is the encoder mask
scores = torch.randn(3, 1)
mask = torch.tensor([[0, -float('inf'), -float('inf')], [0, 0, -float('inf')], [0, 0, 0]])

# Apply the mask to the scores
masked_scores = scores + mask

# Then apply softmax
# attention = F.softmax(masked_scores, dim=-1)

print(scores)
print(mask)

print(masked_scores)