# Constrained Trust Region Policy Optimization

## Setup
For the computational experiments:
```bash
conda create -n ctrpo python=3.9
```
```bash
pip install -r requirements.txt
```

## Reproduce

### Run experiments
By default, the following script will run all algorithms on the full benchmark:
```bash
python run_experiments.py
```
For a dry-run, execute 
```bash
python run_experiments.py --workers 0
```

### Draw plots
To draw plots, run the respective notebook, e.g. `plots/plots_benchmark.ipynb` for the benchmark plots. 
By default, the data loading functions in `plots/helpers.py` will load data from `data/runs`, which we provide in case you don't have the resources or the time to re-run the whole benchmark.

Finally, for the C-NPG toy examples you'll need the [julia](https://julialang.org/downloads/) compiler. 
Once you have the `julia` binary, run the following in the terminal:

```bash
julia toy-example-paper.jl
```

## Acknowledgements
The code in this repo is adapted from [SafePO](https://github.com/PKU-Alignment/Safe-Policy-Optimization).