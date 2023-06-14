import ml_collections
import numpy as np

def get_spiral_default_configs():
  config = ml_collections.ConfigDict()


  # --------------- basic ---------------
  config.dimension = 2
  config.seed = 42
  
  # --------------- model ---------------
  
  config.T = 1.0
  config.beta = 8.0
  config.M = 1.
  config.nu = 2*np.sqrt(config.M**(-1)) + config.Gamma + 0.1 # NON CRITICALLY DAMPED

  config.Sigma_xx_0 = 0.001
  config.Sigma_vv_0 = 0.1


  config.optimizer = 'Adam'
  config.lr = 0.005

  model_configs=None
  return config, model_configs

