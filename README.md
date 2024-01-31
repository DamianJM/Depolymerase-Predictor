# Depolymerase-Predictor (DePP)

<p align="center">
<img src="https://github.com/DamianJM/Depolymerase-Predictor/blob/main/DePP_GUI/DePP_logo.png">
</p>
DePP is a machine learning software package that assists in the identification of bacteriophage depolymerases. Please see our paper for how it works:
Magill, D.J., Skvortsov, T.A. DePolymerase Predictor (DePP): a machine learning tool for the targeted identification of phage depolymerases. _BMC Bioinformatics_ 24, 208 (2023). 
https://doi.org/10.1186/s12859-023-05341-w


# Quick Start
There are three ways to use DePP:
1. Online using <a href="https://timskvortsov.github.io/WebDePP/">WebDePP</a>.
2. Using the DePP app for Windows or Mac: simply download the latest version from the <a href="https://github.com/DamianJM/Depolymerase-Predictor/releases">Releases</a>.
3. Using the command line version (DePP_CLI) - see the instructions below on how to install and use it.

# Installation
The easiest way to install DePP_CLI is by using `conda`. First, copy the repo and navigate to the DePP_CLI folder:
```
git clone https://github.com/DamianJM/Depolymerase-Predictor.git
cd Depolymerase-Predictor/DePP_CLI/
```
Create the DePP_CLI conda environemnt:
```
conda env create -f ./environment.yml
```
Switch to the newly created environment using
```
conda activate depp_cli
```


pharokka.py -i <phage fasta file> -o <output directory> -d <path/to/database_dir> -t <threads>

As of pharokka v1.4.0, if you want extremely rapid PHROG annotations, use --fast:

pharokka.py -i <phage fasta file> -o <output directory> -d <path/to/database_dir> -t <threads> --fast

A standalone version of the software is available both in this repository and on SourceForge which contains the required dependencies. This code and the training set are available so that experienced users can make modifications to refine the model, especially as new sequences become available. We will be updating this package continously as new experimentally verified depolymerases become available.

The intended use of the software is to narrow the selection of potential genes linked to observed depolymerase activity to better direct lab work. It can however be absolutely applied to more complex datasets such as metagenomes. 

An online version of this tool has now been released! This is available via the link below: 

https://timskvortsov.github.io/WebDePP/
