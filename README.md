# gpt

This is my first gpt, I adapted it from Andrej Karpathy's excellent video [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY)  I refactored to separate training from generating and also added configuration.

This is tested and works in both native Windows and WSL2 Ubuntu.

I have Windows, and an AMD Radeon 6800 XT, and I wanted to use it to train a gpt, but Cuda doesn't work with Radeon, and ROCm only works in Linux.

The solution is [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows). So this code targets DirectML but it would be a small change to switch it to cuda, ROCm, or any other pytorch backend.

## Setup
[Install miniconda](https://docs.conda.io/en/latest/miniconda.html) (miniconda specifically is required for DirectML support), then run the following commands:  
```bash
conda env create
conda activate gpt
```

## Training
First you generate model.pt by running the following command: (this takes 1.5 hours on my 6800xt - it might be faster or slower on your hardware)

```bash
python train.py
```

## Generating
Then you can generate text by running the following command:
```bash
python generate.py
```
