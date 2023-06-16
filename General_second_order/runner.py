import time
import types 

import pickle
import numpy as np

import os
import json

import jax.numpy as jnp
import jax.random as random
import jax 
import optax

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from utils.modelization_util import get_score

import models.mlpSecondOrderModel as Model
from models.batch_loss import make_batch_loss
from utils.tests_and_measures import make_variances_and_averages_tests
import util

class Runner():
    def __init__(self,opt):

        self.start_time=time.time()

        batch_size = opt.train_batch_size 
        num_timesteps = opt.num_timesteps

        # define the model
        opt.epsilon_model = Model.epsilon_model_module()
        opt.score = get_score(opt)
        opt.batch_loss = make_batch_loss(opt)
        
        self.savingParamsFile = os.path.join(opt.saving_folder , "parameters.npy")
        self.savingOptimStateFile = os.path.join(opt.saving_folder , "optimizer_state.npy")

        # NOTE preprocessor allows to process opt before saving to remove all elements not of basic types
        self.preprocessor = util.SkipFilter([int, str, float, bool], ["parameters"])
    
    def train_or_retrieve_model(self, opt) :
        key = opt.key
        savingParamsFile = self.savingParamsFile

        if os.path.isfile(savingParamsFile) :
            
            with open(os.path.join(opt.saving_folder, 'opt.txt'), 'r') as f:
                print(util.red(" CAUTION ! : previously saved model params are being loaded, there is no verification that model parameters are the same as those of the command line "))
                opt.__dict__.update(json.load(f))

            with open(savingParamsFile, 'rb') as file:
                opt.parameters = pickle.load(file)

            opt.optimizer = optax.adam(opt.lr)
            with open( self.savingOptimStateFile, 'rb') as file:
                opt.optimizer_state = pickle.load(file)
            
            assert hasattr(opt , "index_iter")
            
            self.training(opt, starting_index = opt.index_iter)
        else :
            # plot the Sigma and mu empirically and theoretically
            make_variances_and_averages_tests(opt)

            opt.parameters = Model.generateParameters(opt)
            
            opt.optimizer = optax.adam(opt.lr)
            opt.optimizer_state = opt.optimizer.init(opt.parameters)

            self.training(opt, starting_index = 0)

        return(opt.parameters)
    
    def training(self, opt, starting_index = 0) :

        batch_loss = opt.batch_loss
        optimizer = opt.optimizer
        writer = SummaryWriter( os.path.join( opt.saving_folder, "tensorboard") )

        i = starting_index
        loss_cache = []
        while i < opt.num_train_iter :

            for batch in opt.train_dataloader :
                batch = batch[0]

                opt.key , = random.split(opt.key, 1)
                value, grads = jax.value_and_grad(batch_loss)(opt.parameters, batch, opt.key)
                loss_cache.append(value)
                if i%500 == 0 :
                    print( "step :", str(i), "   -   loss :", str(np.mean(loss_cache)) , "   -   time since launch :", time.strftime("%H:%M:%S", time.gmtime(time.time() - opt.launch_time)) )
                    writer.add_scalar('Loss/train',np.mean(loss_cache) , i)

                    loss_cache = [] 

                if i%5000 == 0 :
                    print(util.green("Saving progression ..." ))
                    opt.index_iter = i

                    with open(self.savingParamsFile, 'wb') as file:
                        pickle.dump(opt.parameters, file)
                    
                    with open( self.savingOptimStateFile, 'wb') as file:
                        pickle.dump(opt.optimizer_state , file)

                    with open( os.path.join(opt.saving_folder, 'opt.txt'), 'w') as f:
                        json.dump( self.preprocessor.filter(opt.__dict__), f, indent=2)
                    

                    print(util.green("Saving DONE \n" ))

                updates, opt.optimizer_state = optimizer.update(grads, opt.optimizer_state)
                opt.parameters = optax.apply_updates(opt.parameters, updates)
                i+=1
        return(opt.parameters)
    
    @jax.jit
    def deterministic_predictor(opt,batch, i, step_size, parameters, time_indices) :

        """ 
        - batch : shape (batch_size, 2, dim, 1)
        - i : integer
        - step_size : shape (batch_size,)
        - parameters : dict of parameters for score(...)
        """

        Gamma = opt.Gamma
        score = opt.score
        beta = opt.beta
        nu = opt.nu
        M = opt.M

        batch_positions = batch[:,0,:] 
        batch_velocities = batch[:,1,:] 

        score_global = score(parameters, batch_positions, batch_velocities, time_indices[i+1])
        score_x = score_global[:,0,:,None]
        score_v = score_global[:,1,:,None]
        print("score_v",score_v.shape)
        print(score_x.shape)
        print("batch_positions" ,batch_positions.shape)
        print("batch_velocities",batch_velocities.shape)

        print("values of update", -( Gamma*batch_positions + 1.0/M*batch_velocities)*beta/2.*step_size[:,None,None] + \
                                jnp.sqrt(Gamma*beta*step_size[:,None,None])*0 + \
                                ( Gamma*batch_positions*0 + 1./2*Gamma*score_x )*beta*step_size[:,None,None] )

        batch_positions_updated = batch_positions + \
                                -( Gamma*batch_positions + 1.0/M*batch_velocities)*beta/2.*step_size[:,None,None] + \
                                jnp.sqrt(Gamma*beta*step_size[:,None,None])*0 + \
                                ( Gamma*batch_positions*0 + 1.*Gamma*score_x )*beta*step_size[:,None,None]
        batch_velocities_updated = batch_velocities + \
                                (batch_positions - nu*batch_velocities)*beta/2.*step_size[:,None,None] + \
                                jnp.sqrt(M*nu*beta*step_size[:,None,None])*0 + \
                                ( nu*batch_velocities*0 + 1.*M*nu*score_v )*beta*step_size[:,None,None]

        batch = jnp.concatenate( ( batch_positions_updated[:,None,:,:], batch_velocities_updated[:,None,:,:]), axis = 1)
        return(batch)
    

    def generate_deterministic_batch(self, opt ) :
    
        """ 
        - key : PRNG key
        - time_indices : shape (num_timsteps, batch_size) , timsteps are ordered from smallest to biggest
        """
        key = opt.key
        parameters = opt.parameters
        num_timesteps = opt.num_timesteps
        batch_size = opt.batch_size
        M = opt.M

        time_indices = jnp.array( list(range(num_timesteps))*batch_size).reshape(batch_size,num_timesteps).T

        key1, opt.key = random.split(opt.key)
        
        stepSize = util.timeSteps2stepSize(num_timesteps, batch_size)
        batch = random.normal(key1, shape = (batch_size, 2, 2,1))*jnp.array([1,M]).reshape((1,-1,1,1)) # prior distribution
        for i in range(num_timesteps- 2, 0, -1) :

            batch = self.deterministic_predictor(opt, batch, i, stepSize[i], parameters, time_indices )
            batch_positions = batch[:,0,:,:]
            ### test 
            if i%100 == 1 :
                plt.figure(figsize = (4,3))
                plt.scatter(batch_positions[:,0,0], batch_positions[:,1,0], s = 0.2, alpha = 0.7)
                plt.title("step : " + str(i))
                plt.show()
            ###

        return(batch)