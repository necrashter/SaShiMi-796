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


## Cauchy Kernel Benchmark

`S4/cauchy.py` can be run as a standalone script. It will perform the same Cauchy kernel computation using the naive and PyKeOps method, and then compare the results.

Run the following script to get more information about the command line arguments:
```bash
python3 S4/cauchy.py -h
```

If you run the benchmark with a large enough sequence length, the naive method will fail due to out of memory error. PyKeOps, on the other hand, should be able handle this with no problems:
```bash
python3 S4/cauchy.py -l 64000
```

It also reports the maximum difference between the matrices computed by these two methods.
Currently, this value is quite large (~0.001); however, all unit tests that compare these two methods pass.
This might be caused by the random initialization of the inputs in the benchmark.
