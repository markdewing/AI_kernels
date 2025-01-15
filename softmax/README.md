# Softmax

[Wikipedia entry on Softmax](https://en.wikipedia.org/wiki/Softmax_function)

Softmax can easily overflow and most implementations use a "safe" version that subtracts the largest element before exponentiating.

Combined max/normalization ("online") passes for safe softmax paper: https://arxiv.org/abs/1805.02867

Some discussion on making an optimized softmax: https://github.com/SzymonOzog/FastSoftmax
