---
title: Experiment I - Toy
date: 2023-08-09
subject: Neural Fields for SSH Interpolation
subtitle: Experiment I - Toy
short_title: Experiment I - Toy
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: interpolation, neural-networks, neural-fields
abbreviations:
    NerFs: Neural Fields
    OI: Optimal Interpolation
    MLP: Multi-Layer Perceptron
    SSH: Sea Surface Height
---


In this section, we will outline a toy experiment to demonstrate the effectiveness of NerF models for spatiotemporal data related to SSH.



---
## Experimental Setup

### Objective

In this experiment, we are interested in showcasing how well the NerF models can fit a spatiotemporal field from a simulation.
We will vary how many samples we see to partially emulate the scalability of the method.

### Data

We will use the SSH variable from the NATL60.

### Models

We will compare the baseline NerF models: MLP, RFF, and SIREN models.


### Metrics

We will compare the maps produced from the models to the true simulation produced from the NATL60 simulation.


---
## Model Hyperparameters


### GP

### NerFs


---
## Results

### Field Maps

### Statistics

### Pixel Densities

### Power Spectrum

