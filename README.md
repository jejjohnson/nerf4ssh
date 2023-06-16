# Neural Fields for Sea Surface Height Interpolation

**WARNING**: This repo is WIP and is being constructed now. Stay Tuned!

My experiments involving sea surface height (SSH) interpolation using coordinate-based neural networks, i.e. Neural Fields (NerFs).


## Installation


### `conda` - Recommended

```bash
git clone https://github.com/jejjohnson/nerf4ssh.git
cd nerf4ssh
mamba env create -f environments/linux.yaml
```

### `pip`

```bash
pip install "git+https://github.com/jejjohnson/nerf4ssh.git"
```

### `poetry`

```bash
git clone https://github.com/jejjohnson/nerf4ssh.git
cd nerf4ssh
mamba create -n nerf4ssh python=3.10 poetry
poetry install
```


## External Packages

I use quite a few of external packages that I've relegated to their own repo.

**Neural Fields**

I use the [`eqx-nerf`](https://github.com/jejjohnson/eqx-nerf) package has all of the NerF algorithms.

```bash
pip install "git+https://github.com/jejjohnson/eqx-nerf.git"
```

**Trainer**

I use the [`eqx-trainer`](https://github.com/jejjohnson/eqx-trainer) package for the NN Trainer and logging.

```bash
pip install "git+https://github.com/jejjohnson/eqx-trainer.git"
```

**OceanBench**

I use the [`OceanBench`](+https://github.com/jejjohnson/oceanbench) framework for the SSH datasets and metrics.

```bash
brew install g++ cmake eigen boost gsl
pip install "git+https://github.com/jejjohnson/oceanbench.git"
```

## Citations



