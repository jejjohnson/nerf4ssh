---
title: Discussion and Conclusions
date: 2023-08-09
subject: Neural Fields for SSH Interpolation
subtitle: Conclusions
short_title: Conclusions
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
---

% ==============================
% Conclusion:
% - introduced a new class of methods
% - showcased a new SSH dataset with associated metrics
% - we could train a NerF that got decent results JUST from data
% - showed that it was faster (inference)
% - Effectively democratized interpolation methods
% ==============================
% ------------------------------------------------------
% 1) can scale!
% 2) DISCUSSION!!!!
% - relate to 4DVarNet
% - potential extensions!
% ------------------------------------------------------
% ✓ initial condition + 4DVarNet
% ✓ derivative + sensitivity analysis
% ✓ derivative + PDE constraints (PINNs)
% ✓ Bayesian 
% ------------------------------------------------------
% x scale (literature, )
% x interpretation (literature, )
% ------------------------------------------------------

## Discussion

In this work, we introduced Neural Fields as a viable interpolation method for sea surface height. We demonstrated that we were able to match the results of the standard Optimal Interpolation algorithm by training on observations. Not only did we get results that were statistically comparable to the standard DUACS, we also achieve inference speeds and memory constraints that are viable even with a modest computational budget. Neural networks are often only good when we have a large amount of training data available. However, this work has made a case for NerFs coupled with altimetry data as a candidate algorithm for interpolation at scale. 

---
## Limitations

---
## Future Work
% ==============================
% Future Work Points:
% - Interpretability -> Sensitivity Analysis, XAI
% - Predictive Uncertainty -> Stochastic NerF, Flow NerF
% - Physical Sense -> PINNs [cite SWE paper]
% - Noise Sensitivity -> ?
% - MegaScale -> My Future work
% * curriculum learning, pretraining on similations
% hybrid NerFs+GPs
% ==============================
% OBJETIVES
% conditional NerFs - modulations, hypernetworks
% learn priors from observations - functa, gen-NerFs
% embed in bigger systems [4DVarNet]
Aside from the more detailed ablation studies related to noise and sampling density, there is much to be done to make these models more appealing and more trustworthy for the broader community. They lack immediate interpretability because they are fully parameterized machines from data. Techniques such as Explainable AI~\cite{XAI} or Physics Informed Neural Nets~\cite{NerFPINNS} could be of use to understand and constrain the NN predictions. Quantifying the predictive uncertainty is another main future direction, in fact, there are extensions that have been explored in the literature~\cite{NerFStochastic,NerFReg,NerFFlow}. Lastly, we observed that the bulk of the computational limit is training these NNs. Under a computational budget, faster training strategies~\cite{NerFScale} and better hardware would be necessary to make these methods scale to global datasets with billions of data points.
