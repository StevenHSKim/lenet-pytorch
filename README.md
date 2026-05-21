# lenet-pytorch
PyTorch implementation of LeNet-5 based on [LeCun et al. (1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).

## Model Structure
```bash
c1.weight                                 [6, 1, 5, 5]                    150
c1.bias                                   [6]                               6
s2.weight                                 [1, 6, 1, 1]                      6
s2.bias                                   [1, 6, 1, 1]                      6
c3.weight                                 [16, -, 5, 5]                 1,516
c3.bias                                   [16]                             16
s4.weight                                 [1, 16, 1, 1]                    16
s4.bias                                   [1, 16, 1, 1]                    16
c5.weight                                 [120, 16, 5, 5]              48,000
c5.bias                                   [120]                           120
f6.weight                                 [84, 120]                    10,080
f6.bias                                   [84]                             84
output.w                                  [10, 84]                        840
Total Parameters                                                       60,840
```

## Implementation Details
| Paper | Implementation |
|---|---|
| Scaled tanh `1.7159·tanh(2/3·a)` | `lenet_pytorch.py/scaled_tanh()` |
| Subsampling: sum-pool + trainable weight/bias | `lenet_pytorch.py/SubsamplingLayer` |
| C3 partial connectivity (Table I) | `lenet_pytorch.py/C3PartialConv` |
| RBF output layer | `lenet_pytorch.py/RBFLayer` |
| MAP criterion | `loss.py/MAPLoss` |
| Weight init `uniform(-2.4/Fi, 2.4/Fi)` | `utils.py/init_weights()` |

## Results
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/05a938a9-438b-47c1-be8a-09a68659acba" />

| Metric | Paper | Repo |
|---|---|---|
| Test Accuracy | 99.05% | 98.97% |
| Test Error | 0.95% | 1.03% |

## Installation
```bash
conda create -n lenet python=3.11
pip install -r requirements.txt
```

## Usage
```bash
chmod +x experiments.sh
./experiments.sh
```

## Reference
Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, 1998.
