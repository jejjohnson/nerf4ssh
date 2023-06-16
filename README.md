# Neural Fields for Sea Surface Height Interpolation [**WIP!**]

My experiments involving sea surface height (SSH) interpolation using coordinate-based neural networks, i.e. Neural Fields (NerFs).


## Installation


### `conda` - Recommended

```bash
git clone https://github.com/jejjohnson/nerf4ssh.git
cd nerf4ssh
mamba env create -f environments/linux.yaml
```


## External Packages

I use quite a few of external packages that I've relegated to their own repo.

**Neural Fields** - This package has all of the NerF algorithms I use for the experiments.

```bash
pip install "git+https://github.com/jejjohnson/eqx-nerf.git"
```

**Trainer** - This package has the NN Trainer and logging that I use for this experiment.

```bash
pip install "git+https://github.com/jejjohnson/eqx-trainer.git"
```


**OceanBench** - This package has the datasets and metrics that I use for this paper.

```bash
brew install g++ cmake eigen boost gsl
pip install "git+https://github.com/jejjohnson/oceanbench.git"
```

## Citations



