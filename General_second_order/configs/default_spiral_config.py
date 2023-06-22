
# NOTE there seems to be no need to use ml_collection anymore, the parser from argparse library is ultemately used anyways. 
# We could replace ml collections by the parser
import ml_collections
import numpy as np

def get_spiral_default_configs():
  config = ml_collections.ConfigDict()


  # --------------- basic ---------------
  config.dimension = 2
  config.seed = 42
  
  # --------------- model ---------------
  config.T = 1.0
  config.num_timesteps = 2000
  config.beta = 8.0
  config.Gamma = 0.
  config.M = 1.
  config.nu = 2*np.sqrt(config.M**(-1)) + config.Gamma + 0.1 # NON CRITICALLY DAMPED
  config.Sigma_xx_0 = 0.001
  config.Sigma_vv_0 = 0.1

  # --------------- optimizer and loss ---------------
  config.lr = 0.005
  config.optimizer = 'Adam'
  
  # --------------- training & sampling (corrector) ---------------
  config.train_batch_size = 512
  config.num_train_iter = 50000 # make sure we converge

  model_configs=None
  return config, model_configs

