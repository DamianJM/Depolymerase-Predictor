# Depolymerase-Predictor (DePP)

<p align="center">
<img src="https://github.com/DamianJM/Depolymerase-Predictor/blob/main/DePP_GUI/DePP_logo.png">
</p>

DePP is a machine learning software package that assists in the identification of bacteriophage depolymerases. The model is trained on sequences of experimentally verified enzymes. The intended use of the software is to narrow the selection of potential genes linked to observed depolymerase activity to better direct lab work. It can however be absolutely applied to more complex datasets such as metagenomes. Please see our paper for how it works:

Magill, D.J., Skvortsov, T.A. DePolymerase Predictor (DePP): a machine learning tool for the targeted identification of phage depolymerases. _BMC Bioinformatics_ **24**, 208 (2023);
https://doi.org/10.1186/s12859-023-05341-w

A standalone version of the software is available both in this repository and on <a href="https://sourceforge.net/projects/depolymerase-predict/">SourceForge</a> which contains the required dependencies. This code and the training set are available so that experienced users can make modifications to refine the model, especially as new sequences become available. We will be updating this package continously as new experimentally verified depolymerases become available.

# Quick Start
There are three ways to use DePP:
1. Online using <a href="https://timskvortsov.github.io/WebDePP/">WebDePP</a>. Please note that this version can only be used for small scale projects.
2. Using the DePP app for Windows or Mac: simply download the latest version from the <a href="https://github.com/DamianJM/Depolymerase-Predictor/releases">Releases</a>.
3. Using the command line version (`DePP_CLI`) - see the instructions below on how to install and use it.

# Installation
The easiest way to install `DePP_CLI` is by using `conda`. First, copy the repo and navigate to the DePP_CLI folder:
```
git clone https://github.com/DamianJM/Depolymerase-Predictor.git
cd Depolymerase-Predictor/DePP_CLI/
```
Create the DePP_CLI conda environemnt:
```
conda env create -f ./environment.yml
```
To start using `DePP_CLI`, switch to the newly created environment using 
```
conda activate depp_cli
```
You are welcome to test the latest version of `DePP_CLI` by installing and running it manually. `DePP_CLI` has the following dependencies:

```
- python>=3.9
- biopython>=1.77
- numpy>=1.22
- pandas>=1.4
- scikit-learn>=1.1
```
# Usage
To run `DePP_CLI`, you only need to provide an input `FASTA` file containing protein sequences you wish to analyse you and (optionally) the name of the output `CSV` file to save your predictions:

```
depp_cli.py -i <fasta file> -o <CSV file>
```

Type `depp_cli.py -h` to see the help message:
```
usage: depp_cli.py [-h] -i INPUT [-t TRAINING_SET] [-op OUTPUT_PARAMETERS] [-o OUTPUT_PREDICTIONS]

DePolymerase Predictor (DePP)

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Protein FASTA file as input
  -t TRAINING_SET, --training_set TRAINING_SET
                        CSV file containing training set (default: ./TrainingSet/TrainingSet.csv)
  -op OUTPUT_PARAMETERS, --output_parameters OUTPUT_PARAMETERS
                        CSV file to save calculated protein parameters (optional)
  -o OUTPUT_PREDICTIONS, --output_predictions OUTPUT_PREDICTIONS
                        CSV file to save depolymerase predictions (optional)

```

# Get in touch
Contributions, suggestions, feature requests, and bug reports are welcome and appreciated. Please open an issue or contact Tim on t.skvortsov@qub.ac.uk. 
