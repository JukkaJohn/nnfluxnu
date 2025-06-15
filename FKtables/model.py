import numpy as np
import os
import sys
import torch
import typing
from typing import Union
from torch import nn
from torch import Tensor
from fastkernel import read_fk_table, xarr
import lhapdf
import matplotlib.pyplot as plt
import yaml
import copy

def pdfset_prediction(xgrid: torch.Tensor, pdfname: str) -> torch.Tensor:
   """
   Output of the NN.

   Args:
   xgrid (torch.Tensor)
   """   

   p = lhapdf.mkPDF(pdfname, 0)
   xarr = xgrid.detach().numpy()

   f = []
   flux_output = torch.tensor([0, 0, p.xfxQ(14, xarr[0], 10), p.xfxQ(-14, xarr[0], 10)])
    
   for flav in flux_output:
      f.append(flav.reshape(1)) # initialise a lists for each flavour/dimension in the output
   for i, x in enumerate(xarr[1:]):
      flux_output = torch.tensor([0, 0, p.xfxQ(14, x, 10), p.xfxQ(-14, x, 10)])
      for j,flav in enumerate(f):
         flav = torch.cat([flav,flux_output[j].reshape(1)])
         f[j] = flav
   return f 

class MLmodel(nn.Module):
   def __init__(
         self, 
         layers: list[int] = [4,4,2], 
         activation: list[str] = ["tanh", "tanh", "tanh"],
         output_dimension: int = 1) -> None:
      """
      Neural network with linear layers.

      Args:
      datasets (list[str])
      layers (list[int], optional): 
      activation (list[str], optional):
      output_dimension int
      """
      super().__init__()
      

      model = []

      # Preprocessing function layer.
      self.preproc = PreprocessingLayer()

      # Input layer
      model.append(nn.Linear(1, layers[0], bias = True))

      # Add hidden layers.

      def _get_activation_function(i, act):
         if act[i] == "tanh":
            return nn.Tanh()
         elif act[i] == "sigmoid":
            return nn.Sigmoid()
         elif act[i] == "relu":
            return nn.ReLU()
         elif act[i] == "softplus":
            return nn.Softplus()
         else:
            print(f"Activation function {act[i]} not implemented.")
            return None

      for l in range(len(layers)-1):
         act = _get_activation_function(l, activation)
         if act:
            model.extend([act,nn.Linear(layers[l], layers[l+1], bias = True)])
         else:
            model.extend([nn.Linear(layers[l],layers[l+1], bias = True)])
      model.extend([nn.Softplus()])
      
      # Output layer.
      #model.append(nn.Linear(layers[-1], output_dimension, bias = False))
      #self.out = OutputLayer(layers[-1], output_dimension)
      #model.append(self.out)

      self.network = nn.Sequential(*model)

   def forward(self, xgrid: torch.Tensor) -> torch.Tensor:
      """
      Output of the NN.

      Args:
      xgrid (torch.Tensor)
      """
      f = []
      #NNatOne = self.network(torch.tensor(1.0).reshape(-1)) 
      model_output=self.preproc(xgrid[0].reshape(-1))*self.network(xgrid[0].reshape(-1))
      #NN = self.network(xgrid[0].reshape(-1)) - NNatOne
      #model_output = torch.log(1 + torch.exp(NN))
      #model_output = nn.functional.relu(NN)
      #model_output = NN
       
      for flav in model_output:
         f.append(flav.reshape(1)) # initialise a lists for each flavour/dimension in the output
      for i, x in enumerate(xgrid[1:]):
         model_output = self.preproc(x.reshape(-1))*self.network(x.reshape(-1))
         #NN = self.network(x.reshape(-1)) - NNatOne
         #model_output = nn.functional.relu(NN)
         #model_output = NN
         for j,flav in enumerate(f):
            flav = torch.cat([flav,model_output[j].reshape(1)])
            f[j]=flav
      return f 

   def write_lhapdf_grid(self, xgrid: torch.Tensor, path: str, neutrino_flavour: int):
      """
      Write the fitted machine learning model to an LHAPDF grid.

      Args:
      xgrid (torch.Tensor)
      path (str)
      """
      x = xgrid.numpy()

      if neutrino_flavour == 12: pid = 0
      if neutrino_flavour == -12: pid = 1
      if neutrino_flavour == 14: pid = 2
      if neutrino_flavour == -14: pid = 3

      xf1 = x*self.forward(xgrid)[3].detach().numpy()
      xf2 = x*self.forward(xgrid)[1].detach().numpy()
      xf3 = x*self.forward(xgrid)[0].detach().numpy()
      xf4 = x*self.forward(xgrid)[2].detach().numpy()

      with open(path, "w") as f:
         f.write("PdfType: replica\n")
         f.write("Format: lhagrid1\n")
         f.write("---\n")
         for val in x:
            f.write(str(val)+" ")
         f.write("\n")
         f.write("0.1E+001 0.1E+007\n")
         f.write(f"-14  -12  12  14\n") # should store the pid somewhere. self.pid eventuell?
         for val in zip(xf1, xf2, xf3, xf4):
            f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} \n")
            f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} \n")
         f.write("---")

class PreprocessingLayer(nn.Module):
   """
   Layer that implements a preprocessing function to fix 
   the behaviour of the function at high and low x.
   """

   def __init__(self):
      super().__init__()

      self.alpha = nn.Parameter(0.005+torch.rand(1), requires_grad = False)
      self.beta = nn.Parameter(7+torch.rand(1), requires_grad = False)
      self.norm = nn.Parameter(1E13+torch.rand(1), requires_grad = False)

   def forward(self, x):
      # Here the exponents of the preprocessing function are forced to be positive.
      alpha = self.alpha
      #beta = self.beta.clamp(min=1e-3)
      beta = self.beta
      norm = self.norm
      return norm * torch.pow(x,1-alpha) * torch.pow((1-x),beta)

class OneMinusX(nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, x):
      return 1.0 - x

class Dataset():
   """
   """
   def __init__(self,
                setname: str,
                datapath: str,
                genrep: bool = False,
                replica: int = -1,
                training_validation_split: float = 1.0):

      self.datapath = datapath
      self.genrep = genrep
      self.replica = replica

      # Read the yaml file for the dataset:
      self.binwidth = 1 # ich glaube das kann man auch in den Faktor absorbieren.
      with open(f"{datapath}/yamldb/{setname}.yaml") as file:
         yml = yaml.safe_load(file)
         # Read the FK tables.
         self.FKs = []
         for i, FK in enumerate(yml["FK"]):
            print(f"Reading {FK}")
            self.FKs.append(read_fk_table(f"{datapath}/fastkernel/{FK}"))
            self.FKs[i] = self.FKs[i] * self.binwidth 
         # Read the binsize for the FK tables. 
         binsize_filename = yml["fkbinsize"][0]
         self.binsize = self.read_binsize(f"{datapath}/binning/{binsize_filename}")
         # Read the central values and the uncertainties.
         central_val_filename = yml["data"][0]
         uncertainties_filename = yml["uncertainties"][0]
         self.flavour = self.flavour_to_index(yml["flavour"])
         self.central_values, self.mask = self.read_data(f"{datapath}/data/{central_val_filename}")
         self.central_values = self.central_values * self.binwidth 
         self.central_values = self.central_values[self.mask]
         # Read the uncertainties:
         self.uncertainties, _ = self.read_data(f"{datapath}/uncertainties/{uncertainties_filename}")
         self.uncertainties = self.uncertainties 
         self.uncertainties = self.uncertainties[self.mask]
         # Apply the mask also to the FK table:
         for i, FK in enumerate(self.FKs):
            self.FKs[i] = self.FKs[i][self.mask]
         # for now leave this 
         # The FK operation will tell how to treat the FK tables

         # Read factors with which the FK tables can be multiplied.
         self.factors = []
         for factor in yml["factors"]:
            self.factors.append(factor)
         self.fkoperation = yml["fkoperation"]
         if genrep: 
            #self.pseudodata = torch.normal(self.central_values, 
            #                               torch.sqrt(self.central_values))
            self.pseudodata = torch.normal(self.central_values, 
                                           self.uncertainties)

      # Do the training validation split of the indices.
      self.idx_training = torch.randperm(self.central_values.size(0))[:int(self.central_values.size(0)
                                                                     *training_validation_split)]
      self.idx_validation = [index for index,value in enumerate(self.central_values) 
                                       if index not in self.idx_training.tolist()] 

      # Store the central data for training and validation separately.
      self.central_values_training = self.central_values[self.idx_training]
      self.central_values_validation = self.central_values[self.idx_validation]
      # Store the uncertainties for the training and validation split.
      self.uncertainties_training = self.uncertainties[self.idx_training]
      self.uncertainties_validation = self.uncertainties[self.idx_validation]

      print("data")
      print(self.pseudodata)
      # The same split for the pseudodata if replica is generated.
      if genrep:
         self.pseudodata_training = self.pseudodata[self.idx_training]
         self.pseudodata_validation = self.pseudodata[self.idx_validation]

   def prediction(self, f: list[torch.tensor]) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
      if self.fkoperation[0] == "ADD":
         res = (torch.matmul(self.FKs[0]*self.factors[0],f[self.flavour[0]]) 
               + torch.matmul(self.FKs[1]*self.factors[1],f[self.flavour[1]])
               + torch.matmul(self.FKs[2]*self.factors[2],f[self.flavour[2]])
               + torch.matmul(self.FKs[3]*self.factors[3],f[self.flavour[3]])
               ) * self.binsize[self.mask] * 1.16186e-09
      else:
         res = torch.matmul(self.FKs[0]*self.factors[0],f[self.flavour[0]]) * self.binsize[self.mask] * 1.16186e-09
      if res.dim() == 0: 
         res = torch.unsqueeze(res,0)
      res_training = res[self.idx_training]
      res_validation = res[self.idx_validation]
      return res, res_training, res_validation
   

   def flavour_to_index(self, flav: list[int]) -> list[int]:
      out = []
      for iflav in flav:
         if(iflav == "12"):
            out.append(0)
         elif(iflav == "-12"):
            out.append(1)
         elif(iflav == "14"):
            out.append(2)
         elif(iflav == "-14"):
            out.append(3)
         elif(iflav == "16"):
            out.append(4)
         elif(iflav == "-16"):
            out.append(5)
      print(f"Flavour {out}")
      return out


   def read_data(self, setname: str):
      """
      Read the data for a dataset.

      Args:
      setname (str)
      """
      try: 
         with open(setname, "r"):
            data = np.loadtxt(setname, ndmin = 1)
      except FileNotFoundError:
         print(f"Data for dataset {setname} not found in {path}")
         exit(0)
      mask = data > 0.0
      return torch.from_numpy(data).float(), mask

   def read_binsize(self, filename: str):
      """
      Read the binsizes for the FK-table

      Args:
      filename (str)
      """

      try:
         with open(filename, "r"):
            binsize = np.loadtxt(filename, ndmin = 1)
      except FileNotFoundError:
         print(f"File containing the binsize was not found.")
         exit(0)
      return torch.from_numpy(binsize).float()


class Commondata():
   """
   """
   def __init__(self, 
                datasets: list[str], 
                datapath: str = "./data/",
                genrep: bool = False,
                replica: int = -1,
                training_validation_split: float = 1.0):
      self.datasets = []
      self.genrep = genrep
      for i, dataset in enumerate(datasets):
         self.datasets.append(Dataset(dataset, datapath, genrep, replica, training_validation_split=training_validation_split))

   def covmat_from_systematics(self, logcovmat: bool = False) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
      """
      Compute the covariance matrix given the systematics from the commondata.

      Args:
      data (Commondata)
      """
      for i, dataset in enumerate(self.datasets):
         if i == 0: 
            #stat = dataset.central_values
            #stat_training = dataset.central_values_training
            #stat_validation = dataset.central_values_validation
            uncertainties =  dataset.uncertainties
            uncertainties_training = dataset.uncertainties_training
            uncertainties_validation = dataset.uncertainties_validation
            central_values = dataset.central_values
            central_values_training = dataset.central_values_training
            central_values_validation = dataset.central_values_validation
         else:
            #stat = torch.cat([stat, dataset.central_values])
            #stat_training = torch.cat([stat_training, dataset.central_values_training])
            #stat_validation = torch.cat([stat_validation, dataset.central_values_validation])
            uncertainties = torch.cat([uncertainties, dataset.uncertainties])
            uncertainties_training = torch.cat([uncertainties_training, dataset.uncertainties_training])
            uncertainties_validation = torch.cat([uncertainties_validation, dataset.uncertainties_validation])
            central_values = torch.cat([central_values, dataset.central_values])
            central_values_training = torch.cat([central_values_training, dataset.central_values_training])
            central_values_validation = torch.cat([central_values_validation, dataset.central_values_validation])
      if logcovmat: 
         uncertainties = uncertainties / central_values
         uncertainites_training = uncertainties_training /  central_values_training
         uncertainties_validation = uncertainties_validation / central_values_validation
      inv_uncertainties = 1.0 / uncertainties
      inv_uncertainties[inv_uncertainties == np.inf] = 0
      inv_uncertainties_training = 1.0 / uncertainties_training
      inv_uncertainties_training[inv_uncertainties_training == np.inf] = 0
      inv_uncertainties_validation =  1.0 / uncertainties_validation
      inv_uncertainties_validation[inv_uncertainties_validation == np.inf] = 0
      covmat = torch.diag(inv_uncertainties**2)
      covmat_training = torch.diag(inv_uncertainties_training**2)
      covmat_validation = torch.diag(inv_uncertainties_validation**2)
      #covmat_training = torch.diag(1/stat_training)
      #covmat_validation = torch.diag(1/stat_validation)
      return covmat, covmat_training, covmat_validation

   def data_vector(self) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
      for i, dataset in enumerate(self.datasets):
         if i == 0:
            if self.genrep: 
               D = dataset.pseudodata
               D_training = dataset.pseudodata_training
               D_validation = dataset.pseudodata_validation
            else:
               D = dataset.central_values
               D_training = dataset.central_values_training
               D_validation = dataset.central_values_validation
         else:
            if self.genrep:
               D = torch.cat([D, dataset.pseudodata])
               D_training = torch.cat([D_training, dataset.pseudodata_training])
               D_validation = torch.cat([D_validation, dataset.pseudodata_validation])
            else:
               D = torch.cat([D, dataset.central_values])
               D_training = torch.cat([D_training, dataset.central_values_training])
               D_validation = torch.cat([D_validation, dataset.central_values_validation])
      return D, D_training, D_validation
            
   def get_prediction(self, fnu: list[torch.Tensor]) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
      for i, dataset in enumerate(self.datasets):
         if i == 0:
            P, P_training, P_validation = dataset.prediction(fnu)
         else:
            tmp, tmp_training, tmp_validation = dataset.prediction(fnu)
            P = torch.cat([P, tmp])
            P_training = torch.cat([P_training, tmp_training])
            P_validation = torch.cat([P_validation, tmp_validation])
      return P, P_training, P_validation


class Chi2(nn.Module):
   def __init__(self, datasets: list[str], replica: int = -1, training_validation_split: float = 1.0, log_transform: bool = False):
      super(Chi2, self).__init__()
      genrep = replica >= 0
      self.data = Commondata(datasets, genrep=genrep, replica=replica, training_validation_split=training_validation_split)
      self.covmat, self.covmat_training, self.covmat_validation = self.data.covmat_from_systematics()
      self.logcovmat, self.logcovmat_training, self.logcovmat_validation = self.data.covmat_from_systematics(logcovmat = True)
      self.D, self.D_training, self.D_validation = self.data.data_vector()
      self.log_transform = log_transform

   def forward(self, prediction: list[torch.tensor]):
      P, P_training, P_validation = self.data.get_prediction(prediction)
      DP = (self.D - P) 
      DP_training = (self.D_training - P_training) 
      DP_validation = (self.D_validation - P_validation) 
      if self.log_transform:
         P = torch.log(P)
         P_training = torch.log(P_training)
         P_validation = torch.log(P_validation)
         DP = torch.log(self.D) - P
         DP_training = torch.log(self.D_training) - P_training
         DP_validation = torch.log(self.D_validation) - P_validation
         chi2 = torch.dot(torch.matmul(self.logcovmat,DP),DP) / len(P)
         chi2_training = torch.dot(torch.matmul(self.logcovmat_training,DP_training),DP_training) / len(P_training)
         chi2_validation = torch.dot(torch.matmul(self.logcovmat_validation,DP_validation),DP_validation) / len(P_validation)
      else:
         chi2 = torch.dot(torch.matmul(self.covmat,DP),DP) / len(P)
         chi2_training = torch.dot(torch.matmul(self.covmat_training,DP_training),DP_training) / len(P_training)
         chi2_validation = torch.dot(torch.matmul(self.covmat_validation,DP_validation),DP_validation) / len(P_validation)


      # Add the Lagrange Multiplier to enforce positivity. Penalise negative values for the prediction.
      penalty = 0.0
      for pred in prediction:
         penalty = penalty + 1e10*torch.sum(nn.functional.relu(-pred))
      chi2 = chi2 + penalty
      chi2_training = chi2_training + penalty
      chi2_validation = chi2_validation + penalty
      return chi2, chi2_training, chi2_validation

   def chi2flux(self, flux):
       P, P_training, P_validation = self.data.get_prediction(flux)
       P = P.detach().numpy()
       D = self.D.detach().numpy()
       DP = D - P
       covmat = self.covmat.detach().numpy()
       return np.dot(DP,np.matmul(covmat, DP)) / len(P)

   #def DP(self, prediction: list[torch.tensor]):
   #   P, P_training, P_validation = self.data.get_prediction(prediction)
   #   DP = (self.D - P) 
   #   DP_training = (self.D_training - P_training) 
   #   DP_validation = (self.D_validation - P_validation) 
   #   return DP, DP_training, DP_validation

def fit(datasets, 
        layers: list[int] = [4,4,2],
        activation: list[str] = ["none", "none", "none"],
        epochs: int = 2000, 
        max_norm: float = 1.0, 
        minimizer: str = "Adam",
        replica: int = -1,
        training_validation_split: float = 1.0,
        max_counter: int = -1,
        weight_decay: float = 0.0,
        log_transform: bool = False):

   # Initialise the model
   model = MLmodel(layers = layers, activation = activation, output_dimension = 6)
   best_fit_weights = None

   loss_function = Chi2(datasets, replica, training_validation_split, log_transform = log_transform)
   print(model)





   # Set the minimizer.
   if minimizer == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=weight_decay)
      #optimizer = torch.optim.Adam(model.parameters())

   elif minimizer == "SGD":
      #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum = 0.9)
      optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay)
   elif minimizer == "Adadelta":
      optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=weight_decay)
   elif minimizer == "RMSprop":
      optimizer = torch.optim.RMSprop(model.parameters())
   else:
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

   
   # Get the array on which the FK tables are computed.
   xt = torch.from_numpy(np.array(xarr)).float()

   # Print the Chi2 for the neutrino flux:
   flux_lha = pdfset_prediction(xt, "FASERv_Run3_EPOS+POWHEG_7TeV")
   print("Flux chi2")
   print(loss_function.chi2flux(flux_lha))

   # Trainingschleife
   loss_list = []
   loss_training_list = []
   loss_validation_list = []
   best_chi2 = sys.float_info.max # Set it to the largest floating point number.
   counter = 0 
   stopping = max_counter > 0

   for epoch in range(epochs):

      counter = counter + 1

      input_tensor = model(xt)

      loss, loss_training, loss_validation = loss_function(input_tensor)
      if training_validation_split == 1.0: loss_validation  = loss_training

      if loss_validation < best_chi2:
         best_fit_weights = copy.deepcopy(model.state_dict())
         best_chi2 = loss_validation
         chi2total = loss
         counter = 0

      if counter > max_counter and stopping: 
         model.load_state_dict(best_fit_weights)
         return model, best_chi2, chi2total
      
      loss_list.append(loss.item())
      loss_training_list.append(loss_training.item())
      loss_validation_list.append(loss_validation.item())

      if epoch%100 == 0 : 
         print("Epoch:", epoch, "Min. chi2:", min(loss_list), 
               "Min chi2 training:", min(loss_training_list), 
               "Min. validation:", min(loss_validation_list))
      optimizer.zero_grad()
      loss.backward()
      if max_norm > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
      optimizer.step()

   min_chi2 = min(loss_list)
   model.load_state_dict(best_fit_weights)
   return model, best_chi2, chi2total


if __name__ == "__main__":
   
   irep = int(sys.argv[1])
   print(f"Fitting replica {irep}")
   torch.manual_seed(int(irep))
   #torch.autograd.set_detect_anomaly(True)

   # Should move to some input card à la NNPDF
   datasets = ["El_FASERv_Run3_EPOS+POWHEG_7TeV", "Eh_FASERv_Run3_EPOS+POWHEG_7TeV", "theta_FASERv_Run3_EPOS+POWHEG_7TeV"]
   datasets = ["El_FASERv_Run3_EPOS+POWHEG_7TeV", "Eh_FASERv_Run3_EPOS+POWHEG_7TeV"]
   nrep = 1
   nuflav = 14

   # Fit the model

   # LHAPDF comparison 
   xplt = np.logspace(-4,0,100)
   p = lhapdf.mkPDF("faserv", 0)

   # Create figure
   for rep in range(nrep):
      model, chi2, chi2tot = fit(datasets, 
                                 layers = [24, 18, 12, 4],
                                 activation = ["tanh", "relu", "softplus"],
                                 #activation = ["none", "none", "none"],
                                 minimizer = "Adam",
                                 epochs = 10000,
                                 replica = rep, 
                                 training_validation_split = 0.80,
                                 max_counter = 3000,
                                 max_norm = -1.0,
                                 weight_decay = 0.0,
                                 log_transform = False)

      model.write_lhapdf_grid(torch.from_numpy(xplt).float(), 
                              f"faservfit_{irep:04d}.dat", 
                              neutrino_flavour = nuflav)
      with open("chi2list.txt", "a") as chi2list:
         chi2list.write(f"{irep}  {chi2tot}\n")
      irep = irep + 1




