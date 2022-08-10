import sys
sys.path.append("..")
  
import pytest
from pytorch_regression_model.regmodel import RegModel

regmodel = RegModel(input_size=128, hidden_size1=24, hidden_size2=24) 

