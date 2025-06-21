# A brief overview of the NNfluxnu framework

## Equivalence with NNPDF
First of all, the framework is based on the NNPDF framework, which uses feed-forward NNs to fit PDFs from DIS structure functions. The equivalence with NNPDF is exploited in large parts of this framework.

## DIS, neutrino fluxes and event rate measurements
The neutrinos at the LHC are decay products of forward hadron production. These hadrons are produced from pp collisions at ATLAS. The neutrinos go in the forward direction and eventually arrive ~500m downstream the FASER/SND@LHC experiments. At the FASER experiment the neutrino react via deep inelastic scattering using a Tungsten target.
First of all it is important to know how the neutrino fluxes and the event rate measurements are related to each other by DIS variables and kinematic variables.

The neutrinos react via tungsten using deep inelastic scattering like this:
![dis_feynman](img/feynman_dis.png)

Where kinematic variables and the DIS variables are related to each other by:
![dis_var](img/dis_var.png)

In terms of these variables the charged current event rates are related to the neutrino flux by three integrals expressed in this equation:
![events_int](img/events_int.png)
where the integrand is given by
![integrand](img/integrand.png)
where:
![ntlt](img/ntlt.png)
and
![cross_sect](img/cross_sect.png)
is the double differential cross section expressed in terms of DIS structure functions. And finally,
![acc_factor](img/acc_factor.png)


## Fast-Kernel tables: computational efficiency
As can be seen from the equations above, the flux and the event rates are related to each other through several integrals. When trying to parametrise a neutrino flux using an NN this is very computationally expensive, since every new try for the flux results in having to do several integrals. This is why Fast-Kernel are used in this framework to replace the convolutions by a single matrix multiplication. The idea is to take the neutrino flux outside of the integral by expressing it in terms of al inear expansion using Lagrange polynomials. In order to arrive at this result, first the neutrino PDF is defined as :
![neutrinopdf](img/neutrinopdf.png)
Where $x_\nu$ is the momentum fraction and is defined as:
![xnu](img/xnu.png)
and where $s_pp$ is the center of mass from the proton-proton collision. 

The neutrino PDF is then expressed in a linear expansion in an interpolation basis:
![interpolation](img/interpolation.png)

Where $I_\alpha(x_\nu)$ are Lagrange polynomials

Then the number of events are related to the neutrino PDF as follows:
![events_pdf](img/events_pdf.png)

And the FK-table is given by
![fk_table](img/fk_table.png)

And the number of scattering events, the fk-table and the neutrino PDF are then related to each other as follows:
![efficient_events](img/efficient_events.png)

A schematic picture of what happens is displayed below:

![FK_table_visual](img/FK_table.pdf)

## The Machine Learning model
The structure of the machine learning model looks like:

![structure_nn](img/structure_nn.png)

A preprocessing function is imposed to increase computational efficiency, smooth the paramterisation and impose non-divergent behaviour at low and high x. The NN used in the framework contains a few hidden layers with a few nodes. The loss is defined as:
![loss](img/loss.png)

There is also an option to extend the loss to also make sure the neutrino PDF is positive definite when this is not imposed by the activation functions.

When training the NN, a so-called stopping algorithm is employed, which looks like:
![stopping](img/stopping.png)

The idea is to stop training when the loss stops decreasing with a number of patience epochs to prevent the training from stopping too early.

As is the case in NNPDF, the uncertainties on the neutrino PDFs are computed by running so-called closure tests. Using the errors on the data, N Monte Carlo replicas are produced and fitted. Then, the mean and standard deviation is computed using N neutirno PDFs from these N fits. 

The specific parameters used in this work, like the activation function and the optimizer can be found in the fit_settings.yaml file. 

## Hyperoptimization algorithm
An hyperparameter optimization algorithm is also available. The idea behind this algorithm is generality i.e. the methodology should work for all datasets, provided the hyperparameters chosen are sensible. The algorithm in this work uses k-fold cross validation and bayesian optimization to find the best set of hyperparameters. The idea behind k-fold cross validation is to fit the entire dataset consisting of k-folds, except for one. This is then iterated until all possible partitions have been fitted. The hyperparameters which produced the lowest mean loss is then used to run the fit. Bayesian optimization is employed to find this minimum.
## Postfit analysis
The postfit analysis consists of two parts: postift criteria and postfit measures. The first one checks if there is a certain fitted neutrino PDF which did not converge. For example, if the $\chi^2$ from a particular replica deviates more than 4$\sigma$ from the mean $\chi^2$ computed using all the replicas, this replica is discarded. The postfit measures are used to assess the quality of the fit using several statistical measures, such as computing the degree to which the fit has been an overfit/underfit. 


