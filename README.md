# Pointer Network Implementation

This is a PyTorch implementation of the Pointer Network model described in the paper [Pointer Networks](https://arxiv.org/abs/1506.03134) by Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly.

The main contributions of this repository are:
- agile definition of main model, allowing:
  - custom feature extractors (e.g. for images, text, etc.) - just inherit from `FeatureExtractor` and implement the `forward` method
- inference in two permutation modes:
  - [x] permutation with replacement (as in the original paper) - just set `only_uniques` to `False`
  - [x] ***permutation without replacement (new feature)*** - just set `only_uniques` to `True`
- inference in the following stop conditions:
  - [x] fixed number of steps
  - [x] steps equal to the number of elements in the input sequence
  - [ ] model ends when the probability of the stop token is higher than a threshold
- inference with masked inputs, allowing efficient batch processing for sequences of different lengths