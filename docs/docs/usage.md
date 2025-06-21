# How to use the code
In this section a guide on how to use the code is provided including examples.

---

## Data needed to run a fit
First of all, one needs data to run a fit. More specifically one need:

- FK-tables

- Binwidths

- Event rates

- Errors

- Grid nodes

The event rates and errors can either be from event rate measurements or can be sourced from pseudo data. It is also possible to create pseudo data and to rebin the data to a certain number of events if one has: 

- FK-tables

- Binwidths

- Neutrino flux

Using the file generate_data.py one can generate data, rebin it if wanted and write the data to files stored in the Data directory. This data is pseudo data and one needs an input neutrino flux with which event rates can be computed by convoluting this with the FK-table.

All settings for the data generation can be specified in a yaml file like this:
```bash
data:
  pdf: "FASERv_Run3_EPOS+POWHEG_7TeV" 
  min_num_events: 20
  observable: "Eh"
  combine_nu_nub_data: True
  particle_id: 12
  pdf_set: 2
  filename_fk_table: "FK_Eh_final"
  filename_binwidth: "FK_Eh_binsize"
  filename_to_store_events: "FASERv_Run3_EPOS+POWHEG_7TeV_events"
  filename_to_store_stat_error: "FASERv_Run3_EPOS+POWHEG_7TeV_stat_error"
  filename_to_store_sys_error: "FASERv_Run3_EPOS+POWHEG_7TeV_sys_error"
  filename_to_store_cov_matrix: "FASERv_Run3_EPOS+POWHEG_7TeV_cov_matrix"
  division_factor_sys_error: 20
```
Then type:
```bash
python generate_data.py data.yaml
```
To generate data

All the data files should be written to and read from the Data directory.

---

## Running a fit
When one wants to run a fit it starts with a yaml file. In this file all settings are found, for example the structure of the NN, the data one wants to use and the training parameters:

```bash
model:
  hidden_layers: [4, 4,4]
  activation_function: ["softplus","softplus","softplus"]
  preproc: True
  extended_loss: False
  num_output_layers: 1
  num_input_layers: 1
  
closure_test:
  fit_level: 2
  num_reps: 3
  diff_l1_inst: 3

training:
  patience: 100
  max_epochs: 2500
  lr: 0.03
  optimizer: "Adam"
  wd: 0.001
  range_alpha: 5
  range_beta: 20
  range_gamma: 100
  validation_split: 0.0
  max_chi_sq: 5
  lag_mult_pos: 0.001
  lag_mult_int: 0.001
  x_int: [0.001,0.98]

dataset:
  observable: "Eh"
  filename_data: "FASERv_Run3_EPOS+POWHEG_7TeV_events_comb_min_20_events"
  filename_stat_error: "FASERv_Run3_EPOS+POWHEG_7TeV_stat_error_comb_min_20_events"
  filename_sys_error: "FASERv_Run3_EPOS+POWHEG_7TeV_sys_error_comb_min_20_events"
  filename_cov_matrix: "FASERv_Run3_EPOS+POWHEG_7TeV_cov_matrix_comb_min_20_events"
  filename_binning: "FK_Eh_binsize_nub_min_20_events"
  grid_node: 'x_alpha.dat'
  pdf: "FASERv_Run3_EPOS+POWHEG_7TeV"
  pdf_set: 2
  fit_faser_data: False
  
postfit:
  postfit_criteria: True
  postfit_measures: True
  dir_for_data: 'test_dir_faserv_Eh_elec_epos'
  neutrino_pdf_fit_name_lhapdf: 'testgrid'
  particle_id_nu: 12
  particle_id_nub: -12
  produce_plot: True
```

When running a fit type: 
```bash
python execute_fit.py fit_settings.yaml
```
This will perform the fit and also, if wanted, perform the postfit analysis consisting of postfit measures, postfit criteria and plot the result. It will also write the results to a seperate directory and to a separate LHAPDF grid. 

---

## Hyperparameter optimization
An hyperparameter optimization algorithm is also available, based on k-fold cross validation and bayesian optimization. To perform hyperparameter optimizationf for a specific dataset run:

```bash
python perform_hyperopt.py hyperopt_settings.py
```
In the [Framework](framework.md) section, the workings of this algorithm will be explained. 

---