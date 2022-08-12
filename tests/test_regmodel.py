import sys
sys.path.append("..")

import random
import numpy as np  
import torch
import pytest
from pytorch_regression_model.regmodel import RegModel

random_state = 123
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

regmodel = RegModel(input_size=128, hidden_size1=24, hidden_size2=24) 

def test_l1(): 
	parameters = []     
	for parameter in regmodel.parameters(): 
		parameters.append(parameter.view(-1))  
	l1 = regmodel.compute_l1_loss(torch.cat(parameters)) 
	assert np.round(l1.data.item(), 2) == 203.23
