import time
import util

import pickle
import numpy as np

import os

import jax.numpy as jnp
import jax.random as random
import jax 
import optax

import util 
import models.mlpSecondOrderModel as Model
from utils.modelization_util import get_usual_functions
from models.batch_loss import make_batch_loss



class Runner():
    def __init__(self,opt):

        self.start_time=time.time()

        batch_size = opt.train_batch_size 
        num_timesteps = opt.num_num_timesteps

        opt.epsilon_model = Model.epsilon_model_module()

        self.batch_loss = make_batch_loss(opt)

        self.savingFileName = "second_order_parameters_v5" + ("_beta" + str(opt.beta) + "_M" + str(opt.M) + "_nu" + str(opt.nu) + "_Gamma" + str(opt.Gamma)).replace(".","-") +".npy"

    
    def train_or_retrieve_model(self, opt) :
        key = opt.key
        savingFileName = self.savingFileName

        if os.path.isfile(savingFileName) :
            with open(savingFileName, 'rb') as file:
                opt.parameters = pickle.load(file)
        else :
            opt.parameters = Model.generateParameters(opt)

            learning_rate = 0.005
            optimizer = optax.adam(learning_rate)
            opt_state = optimizer.init(parameters)

            loss_cache = []
            i = 0
            while i < opt.num_train_iter :

                for batch in opt.train_dataloader :
                    batch = batch[0]

                    key = random.split(key, 1)[0]
                    value, grads = jax.value_and_grad(self.batch_loss)(parameters, batch, key)
                    loss_cache.append(value)
                    if i%500 == 0 :
                        print( "step :", str(i), "   -   loss :", str(np.mean(loss_cache)))
                        loss_cache = [] 
                    updates, opt_state = optimizer.update(grads, opt_state)
                    parameters = optax.apply_updates(parameters, updates)
                    i+=1

            with open(savingFileName, 'wb') as file:
                pickle.dump(parameters, file)
        return(parameters)