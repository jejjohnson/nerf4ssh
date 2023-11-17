---
title: Introduction
date: 2023-08-09
subject: Neural Fields for SSH Interpolation
subtitle: Introduction
short_title: Introduction
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
---


+++ { "part": "abstract" }

Optimal Interpolation (OI) is a widely used, highly trusted algorithm for interpolation and reconstruction problems in geosciences. With the influx of more satellite missions, we have access to more and more observations and it is becoming more pertinent to take advantage of these observations in applications such as forecasting and reanalysis. With the increase of the volume of available data, scalability remains an issue for standard OI and it prevents many practitioners from effectively and efficiently taking advantage of these large sums of data to learn the model hyperparameters.  Scalability will even more become an issue with the progress of wide-swath satellite observations of ocean surface dynamics (altimetry and currents). Neural Fields (NerFs) are a neural network framework that uses periodic basis functions which have been shown to effectively represent complex signals. They are implicit models that take coordinate-based inputs and output a scalar value or vector field for a variable of interest and have been very successful in many computer vision applications including image regression and 3D rendering. 
In this work, we leverage recent advances in Neural Fields (NerFs) as an alternative to the OI framework where we show how they are intrinsically related and how they can be easily applied to standard reconstruction problems in physical oceanography. We illustrate the relevance of NerFs for gap-filling of sparse measurements of sea surface height (SSH) via satellite altimetry and demonstrate how NerFs are scalable with comparable results to the standard OI. We find that NerFs are a practical set of methods that can be readily applied to geoscience interpolation problems and we anticipate a wider adoption in the future.

+++

---

# Introduction

% ==================================================================
% IDEA - SUMMARY of STATE OF AFFAIRS FOR OCN COMMUNITY
% ==================================================================

**State of Oceanography**. \tofix{The current state of oceanography is great}. On one hand, we have an abundance of observation data that come from satellites, ships and buoys~\tocite{}. On the other hand, our physical knowledge and simulation capabilities have improved drastically due to the development of our understanding of the underlying physical processes that govern the motion of the ocean~\tocite{}. %and it's impact on global climate dynamics~\tocite{}.
%
One large contributing factor to this rapid development is due to the increased processing power from new high performance computing (HPC) environments including CPU, GPU, and TPU~\tocite{something-impressive}.  Greater processing power has allowed the community to effectively increase the speed and resolution of our simulations~\tocite{veros, pytorch,tensorflow,jax, jax-cdf, pdebench}. This has been enhanced with the large community efforts to consolidate our knowledge across different disciplines within the numerical frameworks~\tocite{nemo, mitgcm, cmip5/6, oceanigans,simp-fortran-bridge-thing}. 
%
Another large contributing factor is integrating new data-driven methods, aka statistical learning, machine learning or artificial intelligence, which have allowed us to fit models from the large amounts of observation data alone~\tocite{FNO,WeatherBench}. In addition, the new focus on \textit{physics-informed machine learning}~\tocite{piml-revires} has given us the ability to combine these two sources of information (simulation and observations) with hybrid approaches that improve the effectiveness of the learned models through physics constraints~\tocite{pinns, piml-reviews}.  
%
Ultimately, the combination of all these factors mentioned above (faster computing, improved physical models, and clever learning schemes) has shown to have impressive and pivotal success in many problems in geophysics including forecasting~\tocite{google-weather,nvidia-fno} and interpolation~\tocite{4dvarnet}.

---

% ==================================================================
% IDEA: SSH (DATA) IS A IMPORTANT BUT HARD
% ==================================================================

\textbf{SSH Data Deluge}. Despite the progress on the modeling side, none of this would be possible without the copious amounts of data we are constantly retrieving every day~\tocite{GUS-PAPER}. In oceanography, this data deluge gives them access to many important geophysical variables like sea surface height (SSH), sea surface temperature (SST), sea surface salinity (SSS), ocean color, ocean currents and sea ice. 
%
SSH in particular has gotten special attention with a lot of momentum within the past 50 years. There have been numerous altimetry satellites deployed within the last 10 years~\cite{SSHAltimetry, SSHAltimetry2}~\tocite{diff-altimeters, swot-mission} with increasing frequency and quality. SSH is one the \textit{easiest} variables to accurately measure at scale because we can directly observe it, and thus measure it. It is also a crucial component to derive other important geophysical variables such as ocean currents~\tocite{} and biophysical parameters~\tocite{}. A consequence of the abundance of SSH observations has been the development of high quality, gridded maps of SSH and other derived products~\tocite{copernicus-stuff,mulet-2012,buongiorno-nardelli-2020}. This has resulted in a lot of important downstream studies like mesoscale ocean dynamics~\cite{SSHMesoscale,SSHEDDIES} , biogeochemical transport~\cite{SSHTransport}, and global climate change~\cite{SSHCLIMATE,SSHTransport}.

---

% ==================================================================
% GAP FILLING/INTERPOLATION PROBLEM IS HARD (ESPECIALLY HERE)
% ==================================================================

\textbf{Interpolation Problem}. Despite the increased onset of satellite altimetry, there are still major issues encountered when constructing the derived SSH map products.
%
\textit{ The first problem is that we only ever have noisy, partial observations of SSH through satellite altimetry}. It is impossible for a single satellite altimeter to capture more than XX km$^2$ of coverage on the Earths' surface. If we were to aggregate all altimeters in orbit, we still only cover a small fraction of the Earths' surface. This means that we never have complete coverage of SSH on the Earth at any given moment which results in very sparse observations. The problem is compounded with the fact that the SSH recovered from the altimeters are never perfect.
%
\textit{The second problem is that SSH is a complex, multiscale, dynamical process.} Thus, even theoretically, it would be impossible to confidently interpolate the gaps without knowledge of underlying physical process(es) a priori. The community has many high quality physical models which can be used as priors~\tocite{mitgcm,nemo,veros,oceanigans}. However, many of these models are too expensive, too slow, and/or too cumbersome to run to be able to produce simulations over the entire globe at a satisfactory rate. To reduce the complexity, one can use approximate physical models which capture the most important physics ~\tocite{qg,bfn,sw} however this results in errors which occur from these approximations.
%
\textit{The final problem is that the steady increase in altimetry satellites results in more and more data to process.} Whilst it is true that it is often better to have more data than little data, not all data is informative~\tocite{me :)}. \tofix{A dataset is only informative in relationship to the problem one wishes to solve. For example, a SSH map with an effective spatial resolution of XX km would be useful for X application but it would be useless for X' application. So it is difficult to discern which parts are informative or not.} Generally speaking, a SSH map with a high spatial and temporal resolution would be informative for almost all applications however, one needs to process and learn from TB's of data which renders many algorithms ineffective.
%
So, a model that is able to create a SSH map that can a) interpolate the noisy, partial observations, b) effectively capture the complex, multiscale, dynamics of SSH, and c) scales to TBs or PBs of observation data is a challenge. 


---

% ==================================================================
% CURRENT STATE OF THE LITERATUE (OI, DINEOF, DINAE, BFN, MIOST, 4DVARNET)
% ==================================================================

\textbf{State of Algorithms}. The community has put a lot of research effort into methods that can can interpolate the missing observations which would improve the quality of the SSH map products~\cite{DUACSNEW,SSHOI}. 
%
The earlier methods relied on Objective Analysis (OA) or Optimal Interpolation (OI) schemes~\tocite{Taburet-2019,all-old-papers-DUACS} which is an statistical interpolation scheme which relies on the correlations between neighbouring points within the domain. Despite its age, it is responsible for the current, most widely-used operational product that can produce daily SSH maps at 0.25x0.25 degree resolution~\tocite{copernicus,duacs}. 
%
However, the maps produced by OI can only capture the dynamics up to an effective spatial resolution of XX km which is not enough for many downstream tasks that need to incorporate ocean mesoscale dynamics. Recently, many new algorithms were developed to address this issue and improve the effective resolution from satellite observation data. Some example classes of algorithms include extensions to OI~\tocite{Ardhuin-2020, clement,miost,dymost}, hybrid data assimilation schemes~\tocite{lguenst-2017,florian-2020, benkiran-2021}, and pure data-driven interpolation methods~\tocite{beckers-rixen-2003b,alvera-azcarate-2009, lguensat-2017,tandeo-202,fablet-2020,fablet-chapron-2022,manucharyan-2021,beauchamp-2020}.

However, one of the biggest barriers to the wider adoption of these algorithms from the research field to the operational setting is scale. Many methods have only been demonstrated on local regions with consistent dynamics and relatively small datasets. To the authors knowledge, most of these algorithms have not been applied to a global region with complex dynamics at a greater scale. This feat has only ever been achieved by very few algorithms~\tocite{duacs,miost,dymost} which tend to be simpler. Furthermore, a new mission has started which will increase the amount of observation data by \tofix{ten-fold}~\cite{SWOT,SWOTres}.  It has a wider swath so it will increase our ability to capture more dynamics at a smaller resolution. This will push the limits of even the standard operational methods because they will not be able to process the data due to scale and therefore will not take advantage of the onset of new, high quality data to improve the final SSH map products~\tocite{}.
%
So while there are already plenty of algorithms that exist in the literature, none of them seem to satisfy the requirements of producing good results \textit{and} scale to global data.

---
% ==================================================================
% CONTRIBUTIONS
% ==================================================================
\textbf{Our Contributions}. In this work, we demonstrate an alternative solution to address the computational limitations of interpolation algorithms: neural fields (NerF)~\cite{NeuralFields}. NerFs are a family of coordinate-based, neural network algorithms that learn a function to map a set of spatial-temporal coordinates, $\mathbf{x}$, to a scalar or vector, $\mathbf{y}$, quantity of interest. These methods are parametric versions of the standard OI formulation where the covariance function is replaced by a parameterized NN and the parameters are \textit{learned from data}. 
%
Aside from a few engineering innovations~\cite{NerFSIREN}~\tocite{meta-learning,cyclic-learning}, neural networks are not new algorithms~\tocite{nn-sine-activation} and the ML community has already established the deep connections to GPs~\tocite{infinite-width} and basis functions~\tocite{} (and by extension OI). However, due to the influx of more data from new and improved altimetry satellites~\cite{SWOT,SWOTres}, we hypothesize that these methods are now a feasible option for the oceanography community because we now have enough data to \textit{confidently} learn the parameters. 
%
\textit{In this paper, we demonstrate that NerFs are a viable family of methods that can interpolate the SSH field given partial observations while also being orders of magnitudes faster and memory efficient than the standard OI solution.} We also show that these methods can match the statistical performance of the standard methods via common metrics used in the oceanography community. The ML literature is vast and moves very rapidly so there have already been many new innovations to improve NerFs in practice. Many of these potential improvements are out of scope of this paper, but we also provide an extensive summary of possible extensions based on the applied ML literature.
%
 We are also committed to open-source as we believe it will move the community forward quickly. Thus, all data, preprocessing functions, machine learning modules and training regimes used in this submission are available on \textit{Github}\footnote{\url{https://github.com/jejjohnson/ml4ssh}} with a corresponding set of tutorials. The code is high quality and follows many standard software engineering (SWE) practices that are common within the ML landscape to facilitate replicability, repoducibility and extensibility. We hope that this serves as a solid test bed for future research in interpolating SSH from altimetry data.


---

% ==================================================================
% PAPER OUTLINE
% ==================================================================
\textbf{Paper Outline}. The rest of the paper is organized as follows. In \S2, we give a condensed literature review of methodologies for SSH interpolation within the research community. We state our assumptions about the data representation then we summarize the Gaussian process formulation for our baseline method for Optimal Interpolation. We introduce NerFs as a parametric, data-driven alternative and give the formulation and training procedure. For both methods, we give specific details on how one can use these methods to scale to big data.  In \S3, we outline the interpolation task for an OSSE experiment using NATL60~\tocite{adjayi} simulations and derived pseudo-NADIR and SWOT altimetry tracks. The area of interest (AOI) is over the gulfstream region for a period of $\sim$3 months. This experiment validates how well our method can interpolate realistic sparse observations with known, complex underlying SSH dynamics. We demonstrate how our method compares to other similar methods statistically with metrics like overall normalized RMSE and effective spatial and temporal resolutions. We also showcase how our method performs visually with maps for SSH and its derived physical qoi's (velocity and vorticity). Lastly, we showcase maps in the spectral domain for all physical quantities and we give summary plots for the same physical qoi's. In \S4, we outline the interpolation task for the OSE experiment using real, aggregated altimetry satellite tracks over the gulf stream region for a period of 1 year. This experiment validates how well our method can interpolate realistic sparse observations with unknown, complex underlying SSH dynamics. We largely follow the same validation strategies and visualizations as \S3. In \S5, we outline the interpolation task which extends the OSE experiment in \S4 to global coverage for a period of 5 years. This experiment gives a small demonstration of the scalability and validity of our method to truly big data.  Again, we largely follow the same validation strategies and visualizations as \S3. \S6 gives a discussion of how our method has compared to the standard methods throughout all of the experiments. We also highlight some limitations of the method and suggest some improvements for future work. We end \S6 with some conclusions as well as some next steps.  We have an extensive appendix where we give more details regarding some mathematical formulations, more experimental details, and more figures.