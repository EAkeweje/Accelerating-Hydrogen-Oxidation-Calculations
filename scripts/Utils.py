import numpy as np
# import matplotlib.pyplot as plt
import json
import os

import torch
from sklearn.metrics import r2_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
# from torchvision import transforms
# from torchsummary import summary

##########################Dataset object#########################
class OxidationDataset(Dataset):

  def __init__(self, inputs_path, outputs_dir, nsample, ntimesteps, exclude_h2_o2 = False, exclude_pres = False, return_conc = False):
    """
    Note:
      Deprecated on 27th September
    Args:
        inputs_dir (string): input data path
        outputs_dir (string): output data directory
        nsample (int): data size
        ntimesteps (int, list, or string): number of timesteps in target
        exclude_h2_o2 (bool): To remove H2 and O2 features (from input and output)
        return_conc (bool): To activate conversion of model inputs to concentrations, rather than proportions
    returns:
        Input data tensor
        Output data tensor 
    """
    
    # x = np.loadtxt(inputs_path, skiprows = 1, delimiter= ',')
    # x = np.loadtxt(inputs_path, delimiter= ' ')
    with open(inputs_path, 'rb') as f:
      x = np.load(f)
    self.exclude = exclude_h2_o2
    self.exclude_p = exclude_pres

    #drop inert gases
    if self.exclude:
      self.props = np.delete(x[ :nsample, :], [0, 3, 8, 9], 1)
    else:
      self.props = np.delete(x[ :nsample, :], [8,9], 1)

    self.out_dir = outputs_dir
    self.ntimesteps = ntimesteps
    self.return_conc = return_conc

  def get_total_material(self, Pressure, Temperature):
    R = 8.31441642554361
    return Pressure / (R * Temperature)

  def __len__(self):
    return len(self.props)

  def __getitem__(self, idx):
    #process input
    if self.return_conc:
      #get pressure and temperature
      pres = self.props[idx, -2]
      temp = self.props[idx, -1]
      #compute total amount of all gases
      total_amount_all = self.get_total_material(pres, temp)
      #compute concentrations
      concs = self.props[idx,:-2]*total_amount_all #c_i = x_i * \nu
      input = np.concatenate((concs, self.props[idx,-2:]))
    else:
      input = self.props[idx]

    if self.exclude_p:
      input = np.delete(input, 8)
    #get output
    # Y = np.loadtxt(os.path.join(self.out_dir, f'out{idx}.txt'), delimiter= ';')
    with open(os.path.join(self.out_dir, f'out{idx}.npy'), 'rb') as f:
      Y = np.load(f)

    #drop inert gases
    if type(self.ntimesteps) == int:
      if self.return_conc:
        assert self.ntimesteps != 0, 'Ensure the target time step is different from zeroth timestep'
      target = Y[:self.ntimesteps, 1:-2]
    elif type(self.ntimesteps) == list:
      if self.return_conc:
        assert self.ntimesteps not in self.ntimesteps, 'Ensure the target time steps do not include the zeroth timestep'
      target = Y[self.ntimesteps, 1:-2]
    elif self.ntimesteps == 'all':
      if self.return_conc:
        target = Y[1:, 1:-2]
      else:
        target = Y[1:, 1:-2]
    else:
        raise ValueError("Invalid argument 'ntimestep' should an interger, a list or 'all' string")

    if self.exclude:
      target = np.delete(target, [1,4], axis = 1)

    return torch.from_numpy(input[np.newaxis,:]), torch.from_numpy(target)

#New Dataset object
class ChemDataset(Dataset):

  def __init__(self, inputs_path, outputs_dir, nsample, ntimesteps, in_h2_o2 = True, in_pres = True,
               in_conc = False, out_h2_o2 = True, return_1d_input = False):
    """
    Note:
      Deprecated on 27th September
    Args:
        inputs_dir (string): input data path
        outputs_dir (string): output data directory
        nsample (int): data size
        ntimesteps (int, list, or string): number of timesteps in target
        in_h2_o2 (bool): If false, remove H2 and O2 features from dataset input
        in_pres (bool): If false, remove pressure feature from dataset input and drop zeroth row of target.
        in_conc (bool): If True, activate conversion of inputs to concentrations, rather than molar fractions
        out_h2_o2 (bool): If false, remove H2 and O2 features from dataset target
    returns:
        Input data tensor
        Target data tensor 
    """
    self.in_path = inputs_path
    self.out_dir = outputs_dir
    self.nsample = nsample
    self.ntimesteps = ntimesteps
    self.in_h2_o2 = in_h2_o2
    self.in_pres = in_pres
    self.in_conc = in_conc
    self.out_h2_o2 = out_h2_o2
    self.return_1d_input = return_1d_input

    #load all inputs
    with open(self.in_path, 'rb') as f:
      x = np.load(f)
    #drop inert gases
    self.props = np.delete(x[ :self.nsample, :], [8,9], 1)
    #drop h2 and o2 from input if required
    if not self.in_h2_o2:
      self.props = np.delete(x[ :self.nsample, :], [0, 3], 1)
      
  def get_total_material(self, Pressure, Temperature):
    R = 8.31441642554361
    return Pressure / (R * Temperature)

  def __len__(self):
    return len(self.props)

  def __getitem__(self, idx):
    #process input to concentrations if required
    if self.in_conc:
      #get pressure and temperature
      pres = self.props[idx, -2]
      temp = self.props[idx, -1]
      #compute total amount of all gases
      total_amount_all = self.get_total_material(pres, temp)
      #compute concentrations
      concs = self.props[idx,:-2]*total_amount_all #c_i = x_i * \nu
      input = np.concatenate((concs, self.props[idx,-2:]))
    else:
      input = self.props[idx]

    #remove pressure, if required
    if not self.in_pres:
      input = np.delete(input, -2)

    #get output
    with open(os.path.join(self.out_dir, f'out{idx}.npy'), 'rb') as f:
      Y = np.load(f)

    #obtain target
    ##selectthe required timesteps,
    ##drop time and inert gases.
    if type(self.ntimesteps) == int:
      if self.in_conc:
        assert self.ntimesteps != 0, 'Ensure the target time step is different from zeroth timestep'
      target = Y[:self.ntimesteps, 1:-2]
    elif type(self.ntimesteps) == list:
      if self.in_conc:
        assert 0 not in self.ntimesteps, 'Ensure the target time steps do not include the zeroth timestep'
      target = Y[self.ntimesteps, 1:-2]
    elif self.ntimesteps == 'all':
      if self.in_conc:
        target = Y[1:, 1:-2]
      else:
        target = Y[:, 1:-2]
    else:
        raise ValueError("Invalid argument 'ntimestep' should an interger, a list or 'all' string")

    if not self.out_h2_o2:
      target = np.delete(target, [1,4], axis = 1)

    if self.return_1d_input:
      return torch.from_numpy(input), torch.from_numpy(target).squeeze()
    else:
      return torch.from_numpy(input[np.newaxis,:]), torch.from_numpy(target)

###################for data standardization##################
def standardize(tensor, dim = 0, mean = None, std = None) -> torch.float64:
  '''
  Standardize tensor data,
  Returns standardized tensor, mean and standard deviation
  '''
  tensor = tensor.float()

  if mean is None:
      mean = tensor.mean(dim = dim, keepdim = True)

  if std is None:
      std = tensor.std(dim = dim, keepdim = True)

  standard_tensor = (tensor - mean) / std
  return standard_tensor, mean, std

def inverse_standardize(tensor, dim, mean, std):
    tensor = tensor.float()
    return tensor * std + mean

def load_mean_std(path, loader = None):
  '''
  load mean and std of training data from local disk
  '''
  if os.path.exists(path):
    #get means and stds
    with open(path, 'r') as f:
        mean_std_dict = json.load(f)
    train_x_mean = torch.tensor(mean_std_dict['mean_x'])
    train_x_std = torch.tensor(mean_std_dict['std_x'])
    train_y_mean = torch.tensor(mean_std_dict['mean_y'])
    train_y_std = torch.tensor(mean_std_dict['std_y'])
    return train_x_mean, train_x_std, train_y_mean, train_y_std
  else:
    return save_mean_std(loader, path, output = True)

def save_mean_std(loader, path, output = False):
  '''
  Compute and saves the mean and std of data from dataloader.
  
  loader:: preferably train loader
  path:: a json path
  output:: bool: to output the means and stds
  '''
  #obtaining mean and std of training set
  train_x = []
  train_y = []
  for x,y in loader:
      train_x.append(x)
      train_y.append(y)

  _, train_x_mean, train_x_std = standardize(torch.concat(train_x), 0)
  _, train_y_mean, train_y_std = standardize(torch.concat(train_y), 0)

  #write to disc
  mean_std_dict = {
      'mean_x' : train_x_mean.tolist(),
      'mean_y' : train_y_mean.tolist(),
      'std_x' : train_x_std.tolist(),
      'std_y' : train_y_std.tolist()
  }

  with open(path, 'w') as f:
    json.dump(mean_std_dict, f)
  
  if output:
    return train_x_mean, train_x_std, train_y_mean, train_y_std

##########Training functions#########################
def train_step(model, optimizer, criterion, dataloader, mean_std, device, mb_coeff = 0):
  '''
  Do one training epoch.
  mb_coeff is material balance weight. If mb_coeff == 0, then material balance is not considered.
  '''
  train_loss_ = 0.0
  model.train()
  for input, target in tqdm(dataloader, ascii = True, desc = 'training step'):
    # Transfer Data to GPU if available
    input, target = input.to(device), target.to(device)
    # Standardize
    if mean_std:
      input, _, _ = standardize(input, 0, mean_std[0], mean_std[1])
      target, _, _ = standardize(target, 0, mean_std[2], mean_std[3])
    # Forward Pass
    predict = model(input.float())
    # Find loss
    if mb_coeff != 0:
      #Material Balance
      T_H_true, T_O_true = get_total_H_O_from_fit(target, config['out_h2_o2'])
      T_H_pred, T_O_pred = get_total_H_O_from_fit(predict, config['out_h2_o2'])
      loss = criterion(predict,target.float()) + mb_coeff * criterion(T_H_pred, T_H_true) + mb_coeff * criterion(T_O_pred, T_O_true)
    else:
      loss = criterion(predict, target.float())
    # Clear the gradients
    optimizer.zero_grad()
    # Calculate gradients
    loss.backward()
    # Update Weights
    optimizer.step()
    # Calculate Loss
    train_loss_ += loss.item() 
  return train_loss_

def valid_step(model, criterion, dataloader, mean_std, device, mb_coeff = 0):
  '''
  One validation epoch
  '''
  valid_loss_ = 0.0
  model.eval()     # Optional when not using Model Specific layer
  for input, target in tqdm(dataloader, ascii = True, desc = 'validation step'):
    # Transfer Data to GPU if available
    input, target = input.to(device), target.to(device)
    #standardize
    if mean_std:
      input, _, _ = standardize(input, 0, mean_std[0], mean_std[1])
      target, _, _ = standardize(target, 0, mean_std[2], mean_std[3])
    # Forward Pass
    predict = model(input.float())
    # Find the Loss
    if mb_coeff != 0:
      #Material Balance
      T_H_true, T_O_true = get_total_H_O_from_fit(target, config['out_h2_o2'])
      T_H_pred, T_O_pred = get_total_H_O_from_fit(predict, config['out_h2_o2'])
      loss = criterion(predict,target.float()) + mb_coeff * criterion(T_H_pred, T_H_true) + mb_coeff * criterion(T_O_pred, T_O_true)
    else:
      loss = criterion(predict,target.float())
    # Calculate Loss
    valid_loss_ += loss.item()
  return valid_loss_

def training(model, train_loader, val_loader, config, criterion, mean_std, device):
  '''
  model:: Neural network
  epoch:: 
  optimizer:: optimization algorithm. Default Adam
  learning rate:: 
  dict_path:: path to save best model's state_dict
  criterion:: loss function. Default = nn.MSELoss()
  scheduler:: schedule learning rate. Default = False
  mb_coeff :: Weight for material balance inclusion. Default = 0
  '''
  min_valid_loss = np.inf
  train_losses, val_losses = [], []

  #default weight_decay = 0
  if config['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas = (config['beta1'], config['beta2']), weight_decay = config['weight_decay'])
  elif config['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], momentum= config['SGD_momentum'], weight_decay = config['weight_decay'])
  else:
    raise ValueError(f"Unknown optimizer ({config['optimizer']})")

  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)#old
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.5)
  no_save = 0 #for early stopping

  for e in range(config['epochs']):
    train_loss_ = train_step(model, optimizer, criterion, train_loader, mean_std, device, config['mb_coeff'])
    valid_loss_ = valid_step(model, criterion, val_loader, mean_std, device, config['mb_coeff'])

    train_losses.append(train_loss_)
    val_losses.append(valid_loss_)

    if config['scheduling']:
      scheduler.step()

    if min_valid_loss > valid_loss_:
        min_valid_loss = valid_loss_

        no_save = 0 #reset counter
        # Saving State Dict
        if config['dict_path'] != None:
          torch.save(model.state_dict(), config['dict_path'])

          print(f'Saving Model... Error at Epoch {e+1}: {valid_loss_}')

    else:
      no_save += 1

    # Early stopping
    if no_save >= 200:
      print('Early stopping, no local minimum reach since last 200 epoches')
      break

  return e, min_valid_loss, train_losses, val_losses


######### Material calculations ##############
def get_total_material(Pressure, Temp):
  R = 8.31441642554361
  return Pressure / (R * Temp)


def get_total_H_O_init(input):
  '''
  Computes the total material (moles) of Hydrogen and Oxygen at initiation.
  input: proportion
  '''
  assert input.ndim < 4, f'Tensor dimension must be less than 4: got {input.ndim}'

  if input.ndim == 1:
    T_H = 2*input[0] + input[1] + input[4]+ 2*input[5] + input[6]+ 2*input[7]
    T_O = input[2] + 2*input[3] + input[4]+ input[5] + 2*input[6]+ 2*input[7]

  elif input.ndim == 2:
    T_H = 2*input[:,0] + input[:,1] + input[:,4]+ 2*input[:,5] + input[:,6]+ 2*input[:,7]
    T_O = input[:,2] + 2*input[:,3] + input[:,4]+ input[:,5] + 2*input[:,6]+ 2*input[:,7]

  elif input.ndim == 3:
    T_H = 2*input[:,:,0] + input[:,:,1] + input[:,:,4]+ 2*input[:,:,5] + input[:,:,6]+ 2*input[:,:,7]
    T_O = input[:,:,2] + 2*input[:,:,3] + input[:,:,4]+ input[:,:,5] + 2*input[:,:,6]+ 2*input[:,:,7]
  
  return T_H, T_O


def get_total_H_O_from_fit(input, full):
  assert input.ndim < 4, f'Tensor dimension must be less than 4: got {input.ndim}'
  if full:
    return get_total_H_O_fit_(input)
  else:
    return get_total_H_O_fit(input)

def get_total_H_O_fit_(y):
  '''
  Computes the total material of Hydrogen and Oxygen in full prediction all independents.
  '''
  if y.ndim == 1:
    T_H = 2*y[1] + y[2] + y[5]+ 2*y[6] + y[7]+ 2*y[8]
    T_O = y[3] + 2*y[4] + y[5]+ y[6] + 2*y[7]+ 2*y[8]

  elif y.ndim == 2:
    T_H = 2*y[:,1] + y[:,2] + y[:,5]+ 2*y[:,6] + y[:,7]+ 2*y[:,8]
    T_O = y[:,3] + 2*y[:,4] + y[:,5]+ y[:,6] + 2*y[:,7]+ 2*y[:,8]
  
  elif y.ndim == 3:
    T_H = 2*y[:,:,1] + y[:,:,2] + y[:,:,5]+ 2*y[:,:,6] + y[:,:,7]+ 2*y[:,:,8]
    T_O = y[:,:,3] + 2*y[:,:,4] + y[:,:,5]+ y[:,:,6] + 2*y[:,:,7]+ 2*y[:,:,8]
  
  return T_H, T_O

def get_total_H_O_fit(input):
  '''
  Computes the total material (moles) of Hydrogen and Oxygen in model prediction.
  input: concentration
  '''
  assert input.ndim < 4, f'Tensor dimension must be less than 4: got {input.ndim}'

  if input.ndim == 1:
    T_H = input[1] + input[3]+ 2*input[4] + input[5]+ 2*input[6]
    T_O = input[2] + input[3]+ input[4] + 2*input[5]+ 2*input[6]

  elif input.ndim == 2:
    T_H = input[:,1] + input[:,3]+ 2*input[:,4] + input[:,5]+ 2*input[:,6]
    T_O = input[:,2] + input[:,3]+ input[:,4] + 2*input[:,5]+ 2*input[:,6]

  elif input.ndim == 3:
    T_H = input[:,:,1] + input[:,:,3]+ 2*input[:,:,4] + input[:,:,5]+ 2*input[:,:,6]
    T_O = input[:,:,2] + input[:,:,3]+ input[:,:,4] + 2*input[:,:,5]+ 2*input[:,:,6]
  
  return T_H, T_O


###### dataloader ###################
def make_dataloaders(config):
  '''
  batch_size: int = batch size
  ntimesteps: int or list = number/list of time steps in data
  nsample: int = number of samples to use
  split: list = list of train set to data ration and train+valid set to data ratio
  '''
  #initialize dataset object
  dataset = ChemDataset(inputs_path = config['inputs_path'],
                        outputs_dir = config['outputs_dir'],
                        nsample = config['nsample'],
                        ntimesteps = config['timesteps'],
                        in_h2_o2 = config['in_h2_o2'],
                        in_pres= config['in_pres'],
                        in_conc= config['in_conc'],
                        out_h2_o2 = config['out_h2_o2'],
                        return_1d_input = config['return_1d_input'])
  
  # Creating Training, Validation, and Test dataloaders
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  train_split = int(np.floor(config['split'][0] * dataset_size))
  val_split = int(np.floor(config['split'][1] * dataset_size))
  shuffle_dataset = True
  random_seed = 42
  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices = indices[ : train_split]
  val_indices = indices[train_split : train_split + val_split]
  test_indices = indices[train_split + val_split : ]

  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  test_sampler = SubsetRandomSampler(test_indices)

  train_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=train_sampler, num_workers = 8)
  validation_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=valid_sampler, num_workers = 8)
  test_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=test_sampler, num_workers = 8)
  
  return train_loader, validation_loader, test_loader


#########prediction function #############
def prediction(model, input, target, config, mean_std, device):
  '''
  Make predictions.
  inputs::: 
        tensor of input(s) and target(s)
  returns:::
        tensors of model prediction(s) and target(s)
  '''

  assert input.ndim == target.ndim, f'Input and Target must have same dimensions. Current input and target tensors have {input.ndim} and {target.ndim} dimensions respectively'
  # if target.ndim !=3:
  #   target = target.unsqueeze(0)
  
  #If O2 and H2 are excluded from model dependents
  if not config['out_h2_o2']:
    #Total amount of O and H
    T_H, T_O = get_total_H_O_init(input)
    
    if not config['in_conc']:
      #Total amount
      if input.ndim == 3:
        total_amount = get_total_material(input[:,:, -2], input[:,:,-1])
      elif input.ndim == 2:
        total_amount = get_total_material(input[:, -2], input[:,-1])
      elif input.ndim == 1:
        total_amount = get_total_material(input[-2], input[-1])
      T_H = T_H * total_amount
      T_O = T_O * total_amount

    #Get concentrations of O2 and H2 in target
    if target.ndim == 1:
      C_H2_target = target[1].to(device)
      C_O2_target = target[4].to(device)
      target = np.delete(target, [1,4])
    elif target.ndim == 2:
      C_H2_target = target[:, 1].to(device)
      C_O2_target = target[:, 4].to(device)
      target = np.delete(target, [1,4], axis = 1)
    elif target.ndim == 3:
      C_H2_target = target[:, :, 1].to(device)
      C_O2_target = target[:, :, 4].to(device)
      target = np.delete(target, [1,4], axis = 2)
  
  ax = input.ndim - 1
  #If O2 and H2 are excluded from model independents (features)
  if not config['in_h2_o2']:
    input = np.delete(input, [0,3], axis = ax)

  #If Pressure is excluded from model independents (features)
  if not config['in_pres']:
    input = np.delete(input, -2, axis = ax)

  input, target = input.to(device), target.to(device)
  #transform (standardize) input
  if mean_std:
    input, _, _ = standardize(input, 0, mean_std[0], mean_std[1])
  
  #make prediction
  pred = model(input.float())

  #inverse transform is required. 
  #It makes nosense to compute amount of O and H with transformed data
  if mean_std:
    pred = inverse_standardize(pred, 0, mean_std[2], mean_std[3])

  if not config['out_h2_o2']:
    #get amount of O and H in prediction
    T_H_fit, T_O_fit = get_total_H_O_fit(pred)

    #compute concentration of H2 and O2 for prediction
    C_O2 = 0.5*(T_O.to(device) - T_O_fit)
    C_H2 = 0.5*(T_H.to(device) - T_H_fit)

    #concatenate the concentrations with other predictions
    full_prediction = torch.concat((pred, C_H2.unsqueeze(dim = ax), C_O2.unsqueeze(dim = ax)), dim = ax)
    full_target = torch.concat((target, C_H2_target.unsqueeze(dim = ax), C_O2_target.unsqueeze(dim = ax)), dim = ax)
    return full_prediction, full_target
    
  else:
    return pred, target


####### visuals ##############
def loss_plot(train_loss, valid_loss):
  plt.figure(figsize = (10,7))
  epochs = len(train_loss)
  epochs2 = len(valid_loss)
  plt.plot(range(epochs),train_loss,'-r',label = 'Training loss')
  plt.plot(range(epochs2),valid_loss,'-b',label = 'Validation loss')
  plt.yscale('log')
  plt.legend(fontsize=15)
  plt.grid()
  plt.xlabel('epochs', fontsize=15)
  plt.ylabel('MSE', fontsize=15)
  plt.show()

def show_results(predict, target, idx = 0, lastH2O2 = False):
  '''
  predict::: tensor of predictions
  target::: tensor of targets 
  idx::: index of sample to plot. default = 0
  '''
  assert predict.ndim == target.ndim, 'dimensions of predict and target do not match'

  fig, ax = plt.subplots(3,3, figsize = (20,10))
  if lastH2O2:
    labels = ["T",  'H', 'O', 'OH', 'H2O',  'HO2',  'H2O2', 'H2', 'O2']
  else:
    labels = ["T",  'H2', 'H', 'O', 'O2', 'OH', 'H2O',  'HO2',  'H2O2']
  for i in range(9):
    a,b = i//3, i%3

    if predict.ndim == 2:
      A = predict[:, i]
      A_true = target[:, i]
      ax[a,b].scatter(A.detach().cpu(), A_true.detach().cpu(), alpha = 0.3, label = labels[i])
      ax[a,b].plot([A_true.detach().cpu().min(), A_true.detach().cpu().max()],
        [A_true.detach().cpu().min(), A_true.detach().cpu().max()], 'k')
      ax[a,b].set_ylabel('prediction')
      ax[a,b].set_xlabel('groundtruth')
      ax[a,b].legend(fontsize = 20)

    elif predict.ndim == 3:
      A = predict[idx,:, i]
      A_true = target[idx,:, i]

      ax[a,b].plot(A.detach().cpu(), '-r', label = 'prediction')
      ax[a,b].plot(A_true.detach().cpu(), '-g', label = 'groundtruth')
      ax[a,b].set_title(labels[i])
      ax[a,b].legend()
    
  plt.show()