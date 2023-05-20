# [It's Raw! Audio Generation with State-Space Models](https://arxiv.org/abs/2202.09729)

Karan Goel, Albert Gu, Chris Donahue, Christopher Ré

[*ICML 2022*](https://icml.cc/virtual/2022/poster/17773)

**TODO: Add a simple picture from our results**

This repository provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models.
This re-implementation is provided by:
* İlker Işık, e238051@metu.edu.tr 
* Muhammed Can Keleş, e265013@metu.edu.tr

Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.


## Installation

PyTorch is required. See [PyTorch installation page](https://pytorch.org/get-started/locally/) for more info.
Here's how to install PyTorch with `pip`:
```bash
pip3 install torch torchvision torchaudio
```

Following libraries are required for dataset handling:
```bash
pip3 install numpy scipy
```

**Optional:** [PyKeOps](https://www.kernel-operations.io/keops/index.html) can be installed for more memory-efficient Cauchy kernel computation.
Install PyKeOps using `pip`:
```bash
pip3 install pykeops
```

If that doesn't work, try:
```bash
pip3 install pykeops[full]
```


## Unit Tests

This repository contains numerous unit tests for both S4 and SaShiMi.

Run all unit tests with:
```bash
python3 -m unittest
```

We also have a GitHub Actions Workflow for running these tests.
