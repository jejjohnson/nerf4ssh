---
title: Methods
date: 2023-08-09
subject: Neural Fields for SSH Interpolation
subtitle: Methods
short_title: Methods
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


% ============================
% MAIN POINTS
% ============================
% * Large class of different methods
% * data problems: sparse, noisy, high-dimensional, complex
% * physics-informed - data driven, physics driven, hybrid
% x learning parameters
% * explicit vs implicit (see appendix)
% ---------------------------------------------------
% * implicit
% - covariance based methods (OA,VI,OI,kriging,GPs)
% ---------------------------------------------------
% * explicit methods - gridded techniques
% - Sequential - KF, EnsKF
% - NN - DINOF
% - Knn - Analogue
% - Hybrid - BFN
% - Variational - 4DVar(Net)
% - full field vs sparse
% - noise problem

As mentioned in the introduction, there are many methods in the literature that have been used for SSH reconstruction~\cite{SSHKRIGING,SSH4DVARNET,SSHEOF,SSHDINEOF,SSHDINEOFMulti,SSHDINAE}. However, the most prominent method that is foundational for the Copernicus products is the DUACS algorithm~\cite{DUACSNEW}. The DUACS algorithm is an OI-based scheme which tunes the kernel hyperparameters offline in a physically consistent manner with external altimetry validation datasets. While the standard OI algorithm is an excellent and trusted method for interpolation, this method suffers from scaling issues (training and inference) which can limit its use in practice. The training has a time complexity of $\mathcal{O}(N^3)$ and memory complexity of $\mathcal{O}(N^2)$ which makes this method infeasible without the appropriate hardware~\cite{GPComplexity}. 

To alleviate this issue with scalability, users apply this algorithm locally by only conditioning on the observations available in the local region of interest and interpolating the remaining observations within that region. The algorithm is run along in a patch-like manner until covering the entire globe. This mosaic of many local patches results in a global map of interpolated SSH values. The local application of the OI method has several advantages: 1) It allows the OI framework to be scaled to larger datasets because it reduces the kernel matrix inversion to only the observations within the local region and 2) it removes spurious long-range correlations that are physically inconsistent~\cite{covtaper}.  

In this work, we apply the standard OI method with preconfigured hyperparameters via the local patch-based method described above to serve as the naive baseline which closely mimics the DUACS algorithm. In addition to our naive patch-based OI implementation, we also include the publicly available SSH maps, the solution to the DUACS method, as another baseline method.


---
## Data Representation & Model Assumptions


% - space R^D
% - coordinates (x_s, t)
% - coordinates within a domain, x_s in X, t in T
% - function 
We define spatial coordinate as a vector, $\mathbf{x}_s$, in some spatial domain, $\Omega$, of size $\mathbb{R}^{2}$, which represents the latitude and longitude coordinates. We also define a temporal coordinate as a scalar, $t$, in some temporal domain, $\mathcal{T}$, of size $\mathbb{R}^+$ which represents the continuous time stamp. For an arbitrary geophysical variable, $f$, which is a scalar-valued field, there exists a function, $\boldsymbol{f}$, which maps the spatiotemporal coordinates to the scalar value. This assumption is summarized like so
%
\begin{align}
f&=\boldsymbol{f}(\mathbf{x}_s,t)\equiv\boldsymbol{f}(\mathbf{x}), \hspace{10mm}
\mathbf{x}_s \in \Omega \subset \mathbb{R}^{D_s}, \hspace{10mm} 
t \in \mathcal{T} \subset \mathbb{R}^+, \hspace{10mm} 
\Omega\times\mathcal{T}\times\mathcal{\Theta} \rightarrow\mathbb{R}.
\end{align}
%
In practice, we never observe the true values of the function, $f$. Instead, we observe corrupted versions of this qoi, $y$, and we can assume that they are corrupted by i.i.d. Gaussian noise. 
%
\begin{equation}
y = \boldsymbol{f}(\mathbf{x})+\varepsilon_y, \hspace{10mm} \varepsilon_y\sim \mathcal{N}\left(0,\sigma^2\mathbf{I}\right) \label{eq:regression}
\end{equation}%
Given some observations, e.g. SSH from altimetry satellites, we can aggregate all corresponding observations and coordinate values, $(\mathbf{x},y)$ and create a dataset, $\mathcal{D}=\left\{\mathbf{x}_n,y_n\right\}_{n=1}^N$. The objective is to find some methodology which can approximate the function, $\boldsymbol{f}$, given samples from $\mathcal{D}$. For the remainder of section~\ref{sec:methods}, we will outline two approaches which will prescribe models to approximate this mapping from the observations.


---
## Gaussian Processes


% ================================
% MAIN POINTS
% ================================
% - Standard Methods - OA, VA
% - Other communities - kriging, kernel methods, gaussian processes
% * Discuss formulations
% - We provide the OA formulation
% - We also provide the GP formulation
% * Discuss scaling issues 
% - Discuss current solutions in the literature
% - Discuss our solution in the paper

In the oceanographic community, Optimal Interpolation (sometimes called Objective Analysis) is the most common and has seen the most success in operational settings~\cite{DUACs} and is responsible for the most widely used daily SSH maps which available from the Copernicus website~\tocite{}. 
%
This algorithm assumes that the true field is linear combination of all of the observations, $\{ \mathbf{x}_s\}_{n=1}^N$, covariance matrices and linear Gaussian assumptions has been a staple approach for interpolation satellite observations of SSH. The assumptions are stated clearly, it is robust, and it is a very flexible approach. However, there exists a large body of literature within other communities. For example, the geostatistics community use the term \textit{kriging}~\tocite{kriging-book} which has had a lot of success in the environmental sciences like hydrology~\tocite{hydrology-kriging}. 
%
and the machine learning community has two kernel methods~\tocite{gus-book,sakame} and Gaussian process community~\tocite{summary-paper} which has largely focused on the algorithmic aspects. The two formulations and motivations from the kernel methods and GP are different, but they have been shown to be equivalent for both the standard formulation~\tocite{wild-2021-arxiv,} and sparse/Nystr\"om formulation~\tocite{kanagawa-2018-arxiv}. The kernel methods literature focuses on. In the following sections, we outline the formulation, detail the training regime and talk about some of the limitations with scale.
%

---

### Formulation


In our description of the formulation, we use the GP point of view as it is the most consistent with the equations found in the OI method. The motivations are different but the end result is consistent across all perspectives. We are in the same regression setting as equation~\ref{eq:regression} where we are interested in finding the a function, $\boldsymbol{f}$, that best fits the observations, $y$, at the corresponding spatial-temporal values, $\mathbf{x}$. However, we are interested in finding a distribution over functions, $p(\boldsymbol{f})$, that explain the data. This transforms the standard parametric regression problem into a probabilistic regression problem. Instead of sampling the parameters, we induce a distribution over the functions which allows us to sample the functions directly. However, the function is an infinitely long vector of function values as there are infinite possibilities to explain the data. So one needs to make assumptions over the distribution of function values. 

\textbf{Prior}. GPs are a nonparametric prior distributions over the space of functions and it is fully described by a GP given a mean function and a covariance function (kernel).
%
\begin{align}
 \label{eq:gp-prior} 
&\boldsymbol{f}(\cdot)\sim\mathcal{GP}\left( \boldsymbol{m}(\cdot),\mathbf{K}\right), &&
\boldsymbol{m}(\mathbf{x}): \mathbb{R}^{D_\mathbf{x}}\rightarrow\mathbb{R} &&
 [K]_{ij}=\boldsymbol{k}(\cdot,\cdot') &&
\boldsymbol{k}:\mathbb{R}^{D_\mathbf{x}}\times\mathbb{R}^{D_\mathbf{x}}\rightarrow\mathbb{R}
\end{align}
%
\textbf{Mean Function}. The mean function is the prior belief of the function value at a given location. It acts an an average function of the distribution over functions. This allows us to bias the model for example to encode physical knowledge. For example, we can use a parameterized function (e.g. linear, exponential, neural network) or we can encode a dynamical systems equation~\tocite{}. This can also simplify the learning problem with a biased mean function because then the GPR model is reduced to modeling the noise. In practice, we often use an agnostic zero-mean function when we have an absence of data or prior knowledge.

\textbf{Kernel Function}. The kernel function controls the shape of the function value. This computes the covariances and correlations between the unknown function values by looking at the corresponding inputs. For example, if the input points, $\mathbf{x}_i$ and $\mathbf{x}_j$, are similar within the kernel function, then the function outputs, $\boldsymbol{f}(\mathbf{x}_i$) and $\boldsymbol{f}(\mathbf{x}_j)$, are expected to be similar. This allows us to encode a lot of high-level structural assumptions about the model, e.g. smoothness, periodicity, Brownian motion, etc. In practice, we often use the RBF kernel function
\begin{equation}
\boldsymbol{k}(\mathbf{x},\mathbf{x}')=\sigma^2_k\exp\left(-\frac{||\mathbf{x}-\mathbf{x}'||^2_2}{\lambda_k^2}\right) \label{eq:kernel-rbf}
\end{equation}
which is smooth and infinitely differentiable with two hyperparameters. It is also known as the universal approximator. The length scale, $\ell_k$, is a smoothness parameter which controls how much does one move in the input space before the function value changes significantly. For example, in SSH interpolation, we can assume a 7 day period before the observations become uncorrelated for some dynamical processes. We can also assume that there is 1 degree of latitude/longitude before the observations become uncorrelated. The amplitude, $\sigma_k^2$, controls the vertical magnitude of the function we wish to model.

\textbf{Posterior}. If we are given a set of observations, $\mathcal{D}=\left\{ \mathbf{x}_n,y_n \right\}_{n=1}^N=\left\{\mathbf{X}, \mathbf{y} \right\}$, we want a posterior distribution over functions that explains the data. This is given by Bayes theorem
%
\begin{equation}
p(\boldsymbol{f}(\cdot)|\mathcal{D}) = \frac{p(\mathbf{y}|\boldsymbol{f}(\mathbf{X}))p(\boldsymbol{f}(\cdot))}{p(\mathbf{y}|\mathbf{X})}
\end{equation}
%
where we have the following quantities defined:
%
\begin{align}
&\text{Likelihood}: && \quad\quad p(\mathbf{y}|\boldsymbol{f}(\mathbf{X}))=\mathcal{N}(\boldsymbol{f}(\mathbf{X}),\sigma^2\mathbf{I}) 
\label{eq:gp-likelihood} \\
&\text{Marginal Likelihood}: && \quad\quad  p(\mathbf{y}|\boldsymbol{f}(\mathbf{X}))=\int_{\boldsymbol{f}}p(\mathbf{y}|\boldsymbol{f}(\cdot),\mathbf{X})p(\boldsymbol{f}(\cdot)|\mathbf{X})d\boldsymbol{f}  
\label{eq:gp-mll} \\
&\text{Posterior}: && \quad\quad p(\boldsymbol{f}(\cdot)|\mathcal{D})=\mathcal{GP}\left(\boldsymbol{\mu}_\mathcal{GP}(\cdot), \boldsymbol{\Sigma}_\mathcal{GP}(\cdot, \cdot)\right)
\label{eq:gp-post}
\end{align}
%
Notice how all of the equations are Gaussian distributed which means that we can use simple linear algebra to compute all of the integrals exactly.
GPs can be used for Bayesian inference because we can update our prior using observations.
%

\textbf{Predictive Density}. Because we have Gaussian distributions, we can do marginal conditioning on the "query" points, $\mathbf{x}_*$, and the dataset, $\mathcal{D}$. For predictions, we have two options: we can conditional distribution of the function, $p(\boldsymbol{f}_*|\mathbf{x}_*,\mathcal{D})$, which predicts the expected function or, we can get the conditional distribution of the expected observation, $p(y_*|\mathbf{x}_*,\mathcal{D})$, which predicts what we are likely to observe next. In either case, we get closed form predictive densities for the mean and (co)variance for the expected function. For a given query, $\mathbf{x}_*$, at a spatiotemporal location of interest, we have the equations 
%
\begin{align}
\boldsymbol{\mu}_{\text{GP}*}(\mathbf{x}_*) &= \boldsymbol{m}(\mathbf{x}_*)+\boldsymbol{k}_*^\top (\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\left(\boldsymbol{y} - \boldsymbol{m}(\mathbf{x}_*)\right)
,\label{eq:mugp}\\
\boldsymbol{\sigma}_{\text{GP}*}^2(\mathbf{x}_*) &= k_{**}-
     \boldsymbol{k}_{*}^\top (\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\boldsymbol{k}_{*}, \label{eq:sigmagp}
\end{align}
%
where $[\boldsymbol{k}_*]_i=\boldsymbol{k}(\mathbf{x}_i,\mathbf{x}_*)$ is the cross-covariance between all data points in $\mathcal{D}$ and the query and $[k_{**}]_{ij}=\boldsymbol{k}(\mathbf{x}_i,\mathbf{x}_j)$ is the self-covariance of the query point. The predictive density for the expected observation, $y_*$, has the same mean function but the variance function has an additive noise term from the noise likelihood for the observations, $\sigma^2$. 
%
The standard KF and OI predictive mean and covariance equations for the analysis has parallels to this formulation. Notice how equation~\ref{eq:mugp} and~\ref{eq:sigmagp} are similar to equations \S1 and \S2 in~\tocite{rixen-2000} for OI. In both of those methods, the Kalman gain in this formulation is the term $\boldsymbol{k}_*^\top (\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}$ however, in this case, the covariance matrices are learned from the observations. In addition, the innovation is defined as $\boldsymbol{y} - \boldsymbol{m}(\mathbf{x}_*)$ which is a (potentially) learnable mean function.

---

### Training

In GP regression, we still have to deal with the hyperparameters, $\boldsymbol{\theta_\alpha}$, which come from the mean function, the kernel function and the noise likelihood. In this work, we assumed a zero mean prior for this work, a RBF kernel function with a length scale per spatiotemporal component (i.e. $0.5^\circ$ longitude,  $0.5^\circ$ latitude,  7 days time), and we assume a constant noise level of $0.05$ for the SSH observations. So there are 4 hyperparameters in total, $\boldsymbol{\theta_\alpha}\in\mathbb{R}^4$. We can use the maximum likelihood estimation which involves minimizing the log marginal likelihood.  A GP is a MVN distribution so we can use linear algebra to can write the marginal likelihood in equation~\ref{eq:gp-mll} explicitly has as
%
\begin{equation}
p(y|\mathbf{x}) =\mathcal{N}\left(y|\boldsymbol{m}(\mathbf{x}),\mathbf{K}+\sigma^2\mathbf{I}\right)
\end{equation}%
This is the expected likelihood under the GP prior we specify. We can write this analytically as
%
\begin{equation}
\log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = - \frac{1}{2}\left(\mathbf{y}-\boldsymbol{m}(\mathbf{x}_*)\right)^\top\left( \mathbf{K}+\sigma^2\mathbf{I}\right)^{-1}\left(\mathbf{y}-\boldsymbol{m}(\mathbf{x}_*)\right) - \frac{1}{2}\log|\mathbf{K}+\sigma^2\mathbf{I}|-\frac{N}{2}\log 2\pi
\end{equation}
%
This loss function embodies Occam's razor which does a trade-off between model complexity and overfitting. The first term minimizes the data-fit and the second term minimizes the model complexity. There has been other work exploring alternative, more robust ways to estimate the parameters including MAP estimation, variation inference, and HMC~\tocite{fully-bayesian-gps}.
%
% \begin{equation}

% \end{equation}

\textbf{Advantages}. In the machine learning field, GPs are known to be the golden standard for good, \textit{well calibrated uncertainty}~\tocite{wilson-izmailov-2020}. The variances are higher where there are no observations or when the signal matches the kernel function assumptions and the variances are lower when there are an abundance of observations or when the signal does not match the kernel function assumptions. GPs have interpretable hyperparameters that directly correspond to. \textit{GPs are great interpolators}. They are nonparametric methods which means they are parameterized by the data, i.e. the more data we see, the more complex the model and the better the interpolation.

\textbf{Drawbacks}.\textit{ The GP algorithm does not scale well to really big data}. The standard OI/GP scheme is very costly to store all of the data within memory. The bottleneck is within the matrix inversion step when solving for the weight vector. This is $\mathcal{O}(N^3)$ is computation and $\mathcal{O}(N^2)$ in memory.  There are many existing solutions to try and scale kernel methods to large data. In the kernel methods community, they have methods that use GP. The investigation of sparse GP methods is out of scope for this paper but we believe that future work in this area would make a very interesting and very impactful contribution to the community. \textit{GPs are subject to model misspectification}. A GP can potentially model any field given enough data or a fully specified model. However, we never have infinite observations so we need a. {Standard GP's don't work for non-Gaussian data}. They have problems with modeling heavy-tailed, asymmetric or multi-modal marginal distributions. However, this can be alleviated with modified approximate GPs that exhibit T-Student likelihoods. Furthermore, a more robust inference procedure like Gibbs sampling, Elliptical Slice Sampling, or Hamiltonian Monte Carlo. 


---

## Neural Fields

Neural Fields (NerFs) are coordinate-based neural network models that take in coordinates (e.g. spatial-temporal) and output a scalar or vector value for a variable of interest (e.g. sea surface height, sea surface temperature). This family of neural networks have been very successful in many computer vision applications including image regression and 3D rendering~\cite{NeuralFields}. However, their adoption in geosciences (outside of PINNs) is still at a very early stage, especially in ocean science applications~\cite{NERFHYPER}.  Regardless, there have been a lot of algorithmic developments and they have been crucial in revealing how we represent signals via neural networks~\tocite{functa}. In general, all data are measurements that capture some underlying process. For example, images can be 2D pixel values within an array, 3D shapes can be a mesh or point clouds, and audio can be a discrete value within a vector. However, the ML community is finding that there are some advantages to using coordinate-based representations instead of alternative methods~\tocite{See discussion in appendix}.

Traditionally, the ML community has used a standard Multilayer Perceptron (MLP) with a ReLU activation~\tocite{mescheder-2018, park-2018, chen-2018}. However, the standard MLP tends to fail for more complex signals like images, sound waves and fields which often the types of signals we get with natural and physical signals. The first attempt to address this limitation was the Fourier features transformation followed by any standard neural network, i.e. the Fourier Feature Network (FFN)~\cite{NerFFFN}. Another property of signals that are specific to physical systems is the existence of its derivative and higher order derivatives. The standard FFN cannot model derivatives due the function limitation in the concatenated network. The ReLU activation is monotonically increasing and the derivative is activated at a local point within the function. One inductive bias to encode in the model is to use periodic activation functions (e.g. sinusoidal) can have infinite derivatives and are bounded by 1. This implies that the derivatives exist and we can take as many derivatives through the MLP as possible. The success of SIREN has been seen in many applications including images, 3D shapes, audio and quantities defined by PDEs~\tocite{sitmann-2019,mildenhall-2020,raissi-2019}. Note, the sinusoidal activation function has been around for decades~\cite{SINEACT}~tocite{gallant-1988,sopena-1999,candes-1999,parascandolo-2016,stanley-2007}. In addition, for regression problems (which are often not the focus in ML), other activation functions have been implemented, e.g. tanh. However, only recently~\cite{NerFSIREN} demonstrated the sinusoidal activation with a special initialization scheme for the weights and biases enables the network converge much faster and avoid local minima. They also demonstrated the sine activation's effectiveness on many different applications including images, audio, and PDEs which really highlights this as the universal activation function as a simple go-to which works universally for many problems. This is analogues to the ReLU activation function which is the universal go-to for almost all of DL tasks.
%

%
In the physics informed literature, especially for PINNs and neural PDE solvers, the MLP is the standard method due to the universal approximation theorem~\tocite{pinns-papa}. This has been seen with many ocean-related PDEs like the wave equation~\tocite{}, the shallow water equations~\tocite{} and Navier-Stokes equations~\tocite{}. In general, many studies have used sine activation functions~\tocite{lee-1990,lagaris-1998,he-2000,mai-duy-2003,sirignano-2018,raissi-2019}. However, there were reports of many failures with nonsensical solutions to some PDEs~\tocite{pinns-papa}. It is also very possible that the inclusion of the ReLU activation function is responsible for many failures in training PINNs and that the sine activation function eliminates many of these issues as recent papers do not showcase the same problems~\cite{NerFFlow}. In this work, we use the sine activation function for the MLP (denoted SIREN henceforth) due to its simplicity, the universal applicability, and the existence of derivatives which has all been demonstrated in the literature~\cite{NerFSIREN}. However, a full comparison of NerF variants~\cite{NerFMFN,NerFSIRENMOD} for regression problems in oceanographic systems would be an excellent follow-up study. 

---

#### Multi-Layer Perceptron

The simplest architecture for coordinate-based neural networks is the multi-layer perceptron (MLP).


---

#### Random Fourier Features

---
#### SIREN

The formulation for the~\cite{NerFSIREN} is very simple. This is a standard MLP architecture but it uses the sine function as a periodic activation function. So each layer of the SIREN NN is parameterized as follows:
\begin{equation}
\boldsymbol{\phi}_\ell(\mathbf{x}) = \sin\left( \omega_\ell\left(\mathbf{W}_\ell\mathbf{x}+\mathbf{b}_\ell\right)\right) \label{siren-layer}
\end{equation}where the trainable parameters are $\boldsymbol{\theta}_\ell=\left\{\mathbf{W}_\ell,\mathbf{b}_\ell\right\}$ and the fixed hyperparameters are $\alpha=\left\{\omega_\ell\right\}$. Then, as a traditional MLP, this is composed with with one another to get a SIREN NN:


\begin{align}
\boldsymbol{\phi}(\mathbf{x}) &= 
\mathbf{W}_L\left(
\boldsymbol{\phi}_{L-\ell}\circ\boldsymbol{\phi}_{L-2}\circ \ldots\circ\boldsymbol{\phi}_1 \circ \boldsymbol{\phi}_0
\right)(\mathbf{x})+\mathbf{b}_L \label{siren-network} \\
\end{align}
Notice how the last layer of the SIREN NN is an affine transformation with no scaling or non-linearity.


---
### Training

We take all of the i.i.d. corresponding pairs of spatial coordinate values, $\left\{\mathbf{x}_{s,n}\right\}^N_{n=1}$, the temporal coordinate values, $\left\{t_{n}\right\}^N_{n=1}$, and noisy SSH values, $\left\{y_{\text{obs},n}:=\eta_{n}\right\}^N_{n=1}$, to produce a dataset, $\mathcal{D}=\left\{\mathbf{x}_{s,n}, t_{n}, y_{\text{obs},n}\right\}^N_{n=1}=\left\{\mathbf{x}_{n}, y_{n}\right\}^N_{n=1}$. From the Bayesian formulation, we can write the posterior over the parameters is given by Bayes theorem which relates the conditional probability of the parameters, $\boldsymbol{\theta}$, given the dataset, $\mathcal{D}$:
%
\begin{equation}
p(\boldsymbol{\theta}|\mathcal{D})\propto p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})
\end{equation}
%
From equation~\ref{eq:regression}, we assume our observations, $y$, are continuous and conditioned on the spatiotemporal location, $\mathbf{x}$, so we can assume a Gaussian likelihood. So the mean is described by the parameterized neural network and the noise is i.i.d. and constant:
%
\begin{equation}
p(y|\mathbf{x};\boldsymbol{\theta})=\mathcal{N}(y|\boldsymbol{f_\theta}(\mathbf{x}),\sigma^2).
\end{equation}
%
We put no prior on the parameters , i.e. assume a uniform prior, then the posterior is simply given by the likelihood which results in the \textit{maximum likelihood estimation} (MLE) problem. Because we assumed the samples are assumed to be i.i.d., we get the following minimization problem
%
\begin{equation}
\boldsymbol{\theta}^* =\underset{\boldsymbol{\theta}}{\text{argmin}} \hspace{1mm} \sum_{y,\mathbf{x}\in\mathcal{D}}-\log p(y|\mathbf{x};\boldsymbol{\theta})
\end{equation}
%
which is minimizing the conditional negative log-likelihood. If we assume the noise level is 1, $\sigma^2=1$, then the MLE estimator reduces to the \textit{mean squared error} (MSE):
\begin{equation}
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{D}|}\sum_{n\in\mathcal{D}}(\boldsymbol{f_\theta}(\mathbf{x}) - y)^2
\end{equation}During the training regime, we can take minibatches, $\mathcal{B}=\{\mathbf{x}_b, y_{b} \}_{b=1}^B$ which is a proper subset of the dataset, $\mathcal{B}\cap\mathcal{D}=\mathcal{B}$, to reduce the computational load. This results in the stochastic gradient descent (SGD) algorithm or some variant, e.g. Adam~\tocite{} or AdamW~\tocite{}. Updating the gradients for a smaller subset of the data will allow for faster training and better generalization~\tocite{}. See appendix \todo for more specific details about the training regime.

#### Advantages

\textbf{Heavy Data}. Oceanography data is very heavy. There are often not many realizations and many simulations are often hightly correlated. So we often don't have many realizations of a single scene in space in time. NerFs are inherently interpolators which means that they do not need to have many realizations.

\textbf{Learning}. NerFs give us the ability to learn from data. Note: the MIOST algorithm can also be used in the learning setting as one would define the hyperparameters of the basis function and then learn through observations. Also the DUACs algorithm


#### Challenges

\textbf{Training Time}. NerFs are scalable to implement but they are very expensive to train to truly billion and trillion level. Evaluating a 2D+T field of 720x1040x(365x10) would require 1 billion forward passes through the network to see each of the points just once. Then one would need to do N epochs for training which could easily take trillions of iterations. So the neural network capacity is limited, the training time is limited and the inference time can be limiting. Note: these are not just issues for the NerF method.

\textbf{Training Difficulty}. Ciriculum learning

\textbf{Representation}. Loss function constraint. Bigger function







