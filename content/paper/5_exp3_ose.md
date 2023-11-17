---
title: Experiment II - OSE
date: 2023-08-09
subject: Neural Fields for SSH Interpolation
subtitle: Experiment II - OSE
short_title: Experiment II - OSE
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


In this section, we will outline a OSSE experiment to demonstrate the effectiveness of NerF models for spatiotemporal data related to SSH on a real experimental setting.



---
## Experimental Setup

### Objective

In this experiment, we are interested in showcasing how well the NerF models can fit a spatiotemporal field from real observations when we partially observe the system from alongtrack satellite data.

### Data

We will use the SSH variable from the freely available satellite alongtrack data.

### Models

We will compare the baseline NerF models: MLP, RFF, and SIREN models.
We well compare these models to the true simulation as well as the baseline methods used in operational settings: DUACS, MIOST.


### Metrics

We will compare the maps produced from the models to a leave-on-out satellite track.


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