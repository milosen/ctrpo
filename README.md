# Geometric Methods for Constrained Reinforcement Learning
This repository contains the code for the paper [Embedding Safety into RL: A New Take on Trust Region Methods](https://arxiv.org/abs/2411.02957).

It implements the algorithms C-NPG and C-TRPO and contains code to reproduce the benchmarking experiments from the paper.

## Setup
For the computational experiments:
```bash
conda create -n ctrpo python=3.9
```
```bash
pip install -r requirements.txt
```

## Reproduce

By default, the following script will run all algorithms on the full benchmark:
```bash
python algorithms/c-trpo.py --task SafetyAntVelocity-v1
```

### Draw plots
To draw plots, run the respective notebook, e.g. `plots/plots_benchmark.ipynb` for the benchmark plots. 
By default, the data loading functions in `plots/helpers.py` will load data from `data/runs`, which we provide in case you don't have the resources or the time to re-run the whole benchmark.

Finally, for the C-NPG example you'll need [julia](https://julialang.org/downloads/). 
Once you have the `julia` binary, run the following in the terminal:

```bash
julia c-npg.jl
```

## Acknowledgements
The code in this repo is adapted from [SafePO](https://github.com/PKU-Alignment/Safe-Policy-Optimization).
