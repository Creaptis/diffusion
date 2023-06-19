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

from tqdm import tqdm

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
        self.preprocessor = util.SkipFilter([int, str, float, bool, list], ["parameters"])
    
    def train_or_retrieve_model(self, opt) :
        key = opt.key
        savingParamsFile = self.savingParamsFile

        if os.path.isfile(savingParamsFile) :
            
            with open(os.path.join(opt.saving_folder, 'opt.txt'), 'r') as f:
                print(util.red(" CAUTION ! : previously saved model params are being loaded, there is no verification that model parameters are the same as those of the command line "))
                opt.__dict__.update(json.load(f))

                # NOTE jax PRNK keys are essentially uint32 arrays
                opt.key = np.array(opt.key).astype('uint32')

            with open(savingParamsFile, 'rb') as file:
                opt.parameters = pickle.load(file)

            opt.optimizer = optax.adam(opt.lr)
            with open( self.savingOptimStateFile, 'rb') as file:
                opt.optimizer_state = pickle.load(file)
            assert hasattr(opt , "index_iter")
            
            self.training(opt, starting_index = opt.index_iter + 1)
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
        while i < opt.num_train_iter + 1 :
            for batch in opt.train_dataloader :
                batch = batch[0]

                opt.key , = random.split(opt.key, 1)
                value, grads = jax.value_and_grad(batch_loss)(opt.parameters, batch, opt.key)
                loss_cache.append(value)
                if i%500 == 0 :
                    print( "step :", str(i), "   -   loss :", str(np.mean(loss_cache)) , "   -   time since launch :", time.strftime("%H:%M:%S", time.gmtime(time.time() - opt.launch_time)) )
                    writer.add_scalar('Loss/train',np.mean(loss_cache) , i)

                    loss_cache = [] 

                updates, opt.optimizer_state = optimizer.update(grads, opt.optimizer_state)
                opt.parameters = optax.apply_updates(opt.parameters, updates)
                
                if i%5000 == 0 :
                    print(util.green("Saving progression ..." ))
                    opt.index_iter = i 

                    with open(self.savingParamsFile, 'wb') as file:
                        pickle.dump(opt.parameters, file)
                    
                    with open( self.savingOptimStateFile, 'wb') as file:
                        pickle.dump(opt.optimizer_state , file)

                    with open( os.path.join(opt.saving_folder, 'opt.txt'), 'w') as f:

                        # NOTE important to save current key for reproductability, has to be saved as array however
                        opt.key = np.array(opt.key).astype('uint32').tolist()
                        print("!!!!", self.preprocessor.filter(opt.__dict__))
                        json.dump( self.preprocessor.filter(opt.__dict__), f, indent=2)

                        opt.key = np.array(opt.key).astype('uint32')
                        print("A", opt.key)
                    
                    print(util.green("Saving DONE \n" ))

                i+=1

        return(opt.parameters)
    

    def generate_deterministic_batch(self, opt ) :
    
        """ 
        - key : PRNG key
        - time_indices : shape (num_timsteps, batch_size) , timsteps are ordered from smallest to biggest
        """

        key = opt.key
        parameters = opt.parameters
        num_timesteps = opt.num_timesteps
        batch_size = opt.train_batch_size

        Gamma = opt.Gamma
        score = opt.score
        beta = opt.beta
        nu = opt.nu
        M = opt.M

        @jax.jit
        def deterministic_predictor(batch, i, step_size, parameters, time_indices) :

            """ 
            - batch : shape (batch_size, 2, dim, 1)
            - i : integer
            - step_size : shape (batch_size,)
            - parameters : dict of parameters for score(...)
            """

            batch_positions = batch[:,0,:] 
            batch_velocities = batch[:,1,:] 

            score_global = score(parameters, batch_positions, batch_velocities, time_indices[i+1])
            score_x = score_global[:,0,:,None]
            score_v = score_global[:,1,:,None]
            # print("score_v",score_v.shape)
            # print(score_x.shape)
            # print("batch_positions" ,batch_positions.shape)
            # print("batch_velocities",batch_velocities.shape)

            # print("values of update", -( Gamma*batch_positions + 1.0/M*batch_velocities)*beta/2.*step_size[:,None,None] + \
            #                         jnp.sqrt(Gamma*beta*step_size[:,None,None])*0 + \
            #                         ( Gamma*batch_positions*0 + 1./2*Gamma*score_x )*beta*step_size[:,None,None] )

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

        time_indices = jnp.array( list(range(num_timesteps))*batch_size).reshape(batch_size,num_timesteps).T

        key1, opt.key = random.split(opt.key)
        
        stepSize = util.timeSteps2stepSize(num_timesteps, batch_size)
        batch = random.normal(key1, shape = (batch_size, 2, 2,1))*jnp.array([1,M]).reshape((1,-1,1,1)) # prior distribution

        
        skip_number = 50
        tot_number_imgs = opt.num_timesteps//skip_number

        batches_to_plot = np.zeros(shape=(tot_number_imgs,batch_size,2))
        for i in range(num_timesteps - 1 , 0, -1) :

            batch = deterministic_predictor(batch, i, stepSize[i], parameters, time_indices )
            if i%skip_number == 1 :
                batches_to_plot[i//skip_number] = np.array(batch[:,0,:,0]) # positions, not velocities
        
        print(util.green(" plotting figures showing deterministic predictions ... "))

        ### tests plot
        
        fig, axs = plt.subplots( tot_number_imgs//10, 10, figsize = [ 10*5, 5*tot_number_imgs//10])
        for i in range(batches_to_plot.shape[0]) :
            axs[i//10,i%10].scatter(batches_to_plot[i,:,0],batches_to_plot[i,:,1], s = 0.2, alpha = 0.7)
            axs[i//10,i%10].set_title("step : " + str(i*skip_number))
        ###
        fig.savefig( os.path.join( opt.saving_folder , 'deterministic_predictor.png' ) )

        return(batch)
    

    

    def generate_batch(self, opt) :
        
        """ 
        - key : PRNG key
        - time_indices : shape (num_timsteps, batch_size) , timesteps are ordered from smallest to biggest
        """

        key1, opt.key = random.split(opt.key)
        num_timesteps = opt.num_timesteps 
        batch_size = opt.train_batch_size 
        M = opt.M 
        parameters = opt.parameters 
        Gamma = opt.Gamma
        beta = opt.beta
        nu = opt.nu
        score = opt.score

        # NOTE this is a bit suboptimal to recompile it every time . But for simple tests it might do. Compilation time is negligeable to us.
        @jax.jit
        def predictor(batch, i, step_size, parameters, key) :
            """ 
            - batch : shape (batch_size, 2, dim, 1)
            - i : integer
            - step_size : shape (batch_size,)
            - parameters : dict of parameters for score(...)
            """
            key, subkey = random.split(key)
            
            batch_positions = batch[:,0,...] 
            batch_velocities = batch[:,1,...] 

            w = random.normal(subkey, shape = batch.shape)
            w_x = w[:,0,:,:]
            w_v = w[:,1,:,:]

            score_global = score(parameters, batch_positions, batch_velocities, time_indices[i+1])
            score_x = score_global[:,0,:,None]
            score_v = score_global[:,1,:,None]
            

            batch_positions_updated = batch_positions + \
                                    -( Gamma*batch_positions + 1.0/M*batch_velocities)*beta/2.*step_size[:,None,None] + \
                                    jnp.sqrt(Gamma*beta*step_size[:,None,None])*w_x + \
                                    ( Gamma*batch_positions + Gamma*score_x )*beta*step_size[:,None,None]
            batch_velocities_updated = batch_velocities + \
                                    (batch_positions - nu*batch_velocities)*beta/2.*step_size[:,None,None] + \
                                    jnp.sqrt(M*nu*beta*step_size[:,None,None])*w_v + \
                                    ( nu*batch_velocities + M*nu*score_v )*beta*step_size[:,None,None]

            batch = jnp.concatenate( ( batch_positions_updated[:,None,:,:], batch_velocities_updated[:,None,:,:]), axis = 1)
            return(batch)
        

        time_indices = jnp.array( list(range(num_timesteps))*batch_size).reshape(batch_size,num_timesteps).T
        stepSize = util.timeSteps2stepSize(num_timesteps, batch_size)
        batch = random.normal(key1, shape = (batch_size, 2, 2,1))*jnp.array([1,M]).reshape((1,-1,1,1)) # prior distribution

        ### prediction and plot :
        skip_number = 50
        tot_number_imgs = opt.num_timesteps//skip_number

        batches_to_plot = np.zeros(shape=(tot_number_imgs,batch_size,2))
        for i in range(num_timesteps - 1 , 0, -1) :

            batch = predictor(batch, i, stepSize[i], parameters, opt.key )
            if i%skip_number == 1 :
                batches_to_plot[i//skip_number] = np.array(batch[:,0,:,0]) # positions, not velocities
        
        print(util.green(" plotting figures showing predictions ... "))
        fig, axs = plt.subplots( tot_number_imgs//10, 10, figsize = [ 10*5, 5*tot_number_imgs//10])
        for i in range(batches_to_plot.shape[0]) :
            axs[i//10,i%10].scatter(batches_to_plot[i,:,0],batches_to_plot[i,:,1], s = 0.2, alpha = 0.7)
            axs[i//10,i%10].set_title("step : " + str(i*skip_number))
        
        fig.savefig( os.path.join( opt.saving_folder , 'predictor.png' ) )
        ###

        return(batch)