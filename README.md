# Pointer Network Implementation

This is a PyTorch implementation of the Pointer Network model described in the paper [Pointer Networks](https://arxiv.org/abs/1506.03134) by Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly.

## The main contributions of this repository are:
- agile definition of main model, allowing custom feature extractors (e.g. for images, text, etc.) - just inherit from `FeatureExtractor` and implement the `forward` method
- inference in two permutation modes:
  - [x] permutation with replacement (as in the original paper) - just set `only_uniques` to `False`
  - [x] ***permutation without replacement (new feature)*** - just set `only_uniques` to `True`
- inference in the following stop conditions:
  - [x] fixed number of steps
  - [x] steps equal to the number of elements in the input sequence
  - [ ] model ends when the probability of the stop token is reached
- inference with masked inputs, allowing efficient batch processing for sequences of different lengths

## Examples and use cases:

### 1. [x] Generating permutations without replacement of max length `T` 
- sorting elements of a sequence
- selecting top `T` elements from a sequence
```python
import torch
from pointer_network import MLPFeatureExtractor, PointerNetwork

# Padding element of the sequence for the case of variable length sequences in the batch
pad_element = -1

# Sample batched input sequence for argsort problem
batched_input_sequence = torch.tensor(
    [[10, 2, 33, 14, 5, 1, 1, 8, pad_element, pad_element],
     [10, 11, 6, 13, 15, 9, 12, 14, 7, 8, ]]
)

# Each element in the sequence is a scalar, so we need to add a new dimension to the tensor
batched_input_sequence = batched_input_sequence.unsqueeze(-1).float()

# Attention mask for trimming the sequence, in this case masking is turned off on padded elements
batched_attention_mask = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
)

_, batch_sequence_length, element_size = batched_input_sequence.shape

# Feature extractor for each element in the sequence
mlp_feature_extractor = MLPFeatureExtractor(input_size=element_size,
                                            hidden_size=20,
                                            hidden_layers=2,
                                            output_size=32)

# If we want to sort all elements in the sequence, we need to set max_seq_len >= batch_sequence_length
# If we want to fetch only top T elements, we need to set max_seq_len < batch_sequence_length
ptr_network = PointerNetwork(feature_extractor=mlp_feature_extractor,
                             hidden_size=50,
                             max_seq_len=batch_sequence_length,
                             only_uniques=True)

probabilities, peak_indices = ptr_network.forward(inputs=batched_input_sequence,
                                                  attention_mask=batched_attention_mask)
print(probabilities.shape)  # shape: (2, 10, 10)

print(peak_indices)  # tensor([[5, 6, 1, 4, 7, 0, 3, 2, -1, -1],
                     #         [2, 8, 9, 5, 0, 1, 6, 3, 7, 4]])
print(peak_indices.shape)  # shape: (2, 10)
  
```

### 2. [x] Generating permutations with replacement of max length `T`
- selecting top `T` elements from a sequence, allowing duplicates
```python
import torch
from pointer_network import MLPFeatureExtractor, PointerNetwork

# Padding element of the sequence for the case of variable length sequences in the batch
pad_element = -1

# Sample batched input sequence for argsort problem
batched_input_sequence = torch.tensor(
    [[10, 2, 33, 14, 5, 1, 1, 8, pad_element, pad_element],
     [10, 11, 6, 13, 15, 9, 12, 14, 7, 8, ]]
)

# Each element in the sequence is a scalar, so we need to add a new dimension to the tensor
batched_input_sequence = batched_input_sequence.unsqueeze(-1).float()

# Attention mask for trimming the sequence, in this case masking is turned off on padded elements
batched_attention_mask = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
)

_, batch_sequence_length, element_size = batched_input_sequence.shape

# Feature extractor for each element in the sequence
mlp_feature_extractor = MLPFeatureExtractor(input_size=element_size,
                                            hidden_size=20,
                                            hidden_layers=2,
                                            output_size=32)

# Select top T elements from the sequence
max_seq_len = 4
ptr_network = PointerNetwork(feature_extractor=mlp_feature_extractor,
                             hidden_size=50,
                             max_seq_len=max_seq_len,
                             only_uniques=False)

probabilities, peak_indices = ptr_network.forward(inputs=batched_input_sequence,
                                                  attention_mask=batched_attention_mask)
print(probabilities.shape)  # shape: (2, 4, 10)

print(peak_indices)  # tensor([[5, 6, 1, 5],
                     #         [2, 8, 8, 5]])
print(peak_indices.shape)  # shape: (2, 4)
  
```

### 3. [] Generating permutations with replacement and stop condition
- selecting elements from a sequence, allowing duplicates, and stopping when stop token is reached (e.g. Delaunay triangulation)

### 4. [] Generating permutations without replacement and stop condition
- sorting elements of a sequence and stopping when stop token is reached (e.g. TSP, convex hull, etc.)
