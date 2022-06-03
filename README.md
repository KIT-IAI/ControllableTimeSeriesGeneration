# Controlling Non-Stationarity and Periodicities in Time Series Generation Using Conditional Invertible Neural Networks

This repository contains the Python implementation of the approach to approach to control non-stationarity and periodicities with calendar and statistical information when generating time series. The approach is presented in the following paper:
>B. Heidrich, M. Turowski, K. Phipps, K. Schmieder, W. Süß, R. Mikut, and V. Hagenmeyer, 2022, "Controlling Non-Stationarity and Periodicities in Time Series Generation Using Conditional Invertible Neural Networks," in Applied Intelligence, doi: [10.1007/s10489-022-03742-7](https://doi.org/10.1007/s10489-022-03742-7).


## Installation

Before the propsed approach can be applied using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Set up a virtual environment using e.g. venv (`python -m venv venv`) or Anaconda (`conda create -n env_name`). Afterwards, install the dependencies with `pip install -r requirements.txt`. 

### 2. Download Data (optional)

If you do not have any data available, you can download exemplary data by executing `python download.py`. 
This script downloads and unpacks the [ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)  
and the [Bike Sharing Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip) 
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) as CSV files.


## Existing Pipelines

This repository contains the `ablation_study_pipeline.py` for evaluating the influence of the different conditional
information and loss function.

To control non-stationarity and periodicities when generating time series, this repository also contains the
`controllable_generation_pipeline.py`

### Input

See in both pipelines for a description of the arguments. If you would like to use your own dataset you might have to
specify some of these arguments. Moreover, you have to ensure that the index is named `time` and that the delimiter
of your csv file is `,`

### Output

Both pipelines creates result folders, which contains visualisation and summary files containing the evaluation results.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Program “Energy System Design”, the Helmholtz Metadata Collaboration, and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".

## License

This code is licensed under the [MIT License](LICENSE).
