# Welcome to NNfluxnu

The NNfluxnu package is a code based on NNPDF to fit neutrino PDFs using neutrino scattering events at the LHC. Using so-called closure tests, one can parametrise a neutrino flux using a feed-forward NN to make a theory agnostic parametrisation using pseudo data or event rate measurements from FASER/SND@LHC. 

## Installation
First make a conda environment with python=3.10
```bash
conda create -n nnfluxnu python=3.10
conda activate nnfluxnu
```
The code can directly be installed from the git repository using:
```bash
git clone https://github.com/JukkaJohn/nnfluxnu.git
cd nnfluxnu
poetry install
```
The LHAPDF library could not be added to poetry, so this should be added seperately by running:
```bash
conda install -c conda-forge lhapdf
```
This will create a folder in conda_dir/envs/nnfluxnu/share/LHAPDF
All neutrino PDFs in the LHAPDF format should be read and written to here.
So:
```bash
cp -r LHAPDF_data/* conda_path/envs/nnfluxnu/share/LHAPDF
```    

<!-- * `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit. -->



## Project Description
This code was written as part of a Master's project at the University of Amsterdam, Vrije Universiteit Amsterdam and Nikhef. 

In this work, theory agnostic parametrisations of neutrino fluxes are made using feed-forward neutrino fluxes. The paper based on this code and project can be read here (insert link). In order to fit the neutrino fluxes, actual DIS charged current event rate measurements from the FASER collaboration are used as well as pseudo data generated using several MC event generators simulating forward hadron production at the LHC. The neutrino flux is related to the event rate measurements by several integrals. These are replaced by so-called FK-tables which encapsulate all the information on DIS structure functions and are contained in a matrix. These FK-tables improve the computational efficiency significantly. This code can fit both pseudo data and event rate measurements from faser to fit electron and muon neutrinos. All the neutrino fluxes parametrised used in this work are written to LHAPDF grids and can also be found in the git repository. The code was also used to research several physics applications by comparing neutrino fluxes from different observables, MC generators, detector geometries, enhanced BSM decays and the influence of IC on neutrino fluxes. 

The structure of the code can be found in the [Framework](framework.md) section. How one should use the code is explained in the [Usage](usage.md) section. 