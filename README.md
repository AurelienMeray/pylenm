# PyLEnM

[![PyPI version](https://badge.fury.io/py/pylenm.svg)](https://badge.fury.io/py/pylenm)
[![Documentation Status](https://readthedocs.org/projects/pylenm/badge/?version=latest)](https://pylenm.readthedocs.io/en/latest/?badge=latest)


This package aims to provide machine learning (ML) functions for performing comprehensive soil and groundwater data analysis, and for supporting the establishment of effective long-term monitoring. The package includes unsupervised ML for identifying the spatiotemporal patterns of contaminant concentrations (e.g., PCA, clustering), and supervised ML for evaluating the ability of estimating contaminant concentrations based on in situ measurable parameters, as well as the effectiveness of well configuration to capture contaminant concentration distributions. Currently, the main focus is to analyze historical groundwater datasets and to extract key information such as plume behaviors and controlling (or proxy) variables for contaminant concentrations (Schmidt et al., 2018). This is setting a ground for integrating new technologies such as in situ sensors, geophysics and remote sensing data. 

This development is a part of the Advanced Long-Term Monitoring Systems (ALTEMIS) project. In this project, we propose to establish the new paradigm of long-term monitoring based on state-of-art technologies – in situ groundwater sensors, geophysics, drone/satellite-based remote sensing, reactive transport modeling, and AI – that will improve effectiveness and robustness, while reducing the overall cost. In particular, we focus on (1) spatially integrative technologies for monitoring system vulnerabilities – surface cap systems and groundwater/surface water interfaces, and (2) in situ monitoring technologies for monitoring master variables that control or are associated with contaminant plume mobility and direction. This system transforms the monitoring paradigm from reactive monitoring – respond after plume anomalies are detected – to proactive monitoring – detect the changes associated with the plume mobility before concentration anomalies occur.

The latest package can be downloaded from: https://pypi.org/project/pylenm/

More information on the project can be found here: https://altemis.lbl.gov/ai-for-soil-and-groundwater-contamination/ 

--
# Journal Publication:
## [PyLEnM: A Machine Learning Framework for Long-Term Groundwater Contamination Monitoring Strategies](https://pubs.acs.org/doi/full/10.1021/acs.est.1c07440)
Aurelien O. Meray, Savannah Sturla, Masudur R. Siddiquee, Rebecca Serata, Sebastian Uhlemann, Hansell Gonzalez-Raymat, Miles Denham, Himanshu Upadhyay, Leonel E. Lagos, Carol Eddy-Dilek, and Haruko M. Wainwright
Environmental Science & Technology 2022 56 (9), 5973-5983
DOI: 10.1021/acs.est.1c07440

--

# Demonstration notebooks:
[1 – Basics](https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/notebooks/1%29%20pyLEnM%20-%20Basics.ipynb)<br>
[2 - Unsupervised learning](https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/notebooks/2%29%20pyLEnM%20-%20Unsupervised%20Learning.ipynb)<br>
[3 – Water Table Estimation & Well Optimization](https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/notebooks/3%29%20pyLEnM%20-%20Water%20Table%20Spatial%20Estimation%20&%20Well%20Optimization.ipynb)<br>
[4 – Tritium Spatial Estimation](https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/notebooks/4%29%20pyLEnM%20-%20Tritium%20Spatial%20Estimation.ipynb)<br>
[5 – Proxy Estimation (SC~Tritium)](https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/notebooks/5%29%20pyLEnM%20-%20Proxy%20Estimation%20(SC~Tritium).ipynb)<br>

--

# Demonstration data:
The data used in the demonstration notebooks above can be downloaded [here]( https://github.com/ALTEMIS-DOE/pylenm/tree/master/notebooks/data).

--

# Contributors:
Aurelien Meray<br>
Haruko Wainwright<br>
Himanshu Upadhyay<br>
Masudur Siddiquee<br>
Savannah Sturla<br>
Nivedita Patel<br>