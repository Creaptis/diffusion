import numpy as np

from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import util

import os

from utils.saver_loader import load_opt, save_opt

def make_variances_and_averages_tests(opt) :
    # Test : does the forward give the same covariance as predicted by our formulas 
    experimental_batch_size = 1000 # TODO incorporate in opt
    beta = opt.beta
    Gamma = opt.Gamma
    nu = opt.nu
    M = opt.M
    Sigma_xx_0 = opt.Sigma_xx_0
    Sigma_vv_0 = opt.Sigma_vv_0
    dimension = opt.dimension
    batch_size = opt.train_batch_size
    num_timesteps = opt.num_timesteps

    mu_global_HSM  = opt.mu_global_HSM   
    Sigma_xx = opt.Sigma_xx  
    Sigma_vv = opt.Sigma_vv  
    Sigma_xv = opt.Sigma_xv  

    def forward_step(positions, stepSize, key) :
        """ 
        positions : (experimental_batch_size,2,2)
        stepSize : (experimental_batch_size)
        """ 
        positions = positions.reshape((experimental_batch_size,2,2))
        # TODO : accomodate for when f and G depend on time through beta
        f = jnp.array([[-Gamma,M**(-1) ], [ -1, -nu]])*beta/2
        G = jnp.array([[jnp.sqrt(Gamma*beta),0 ], [ 0 , jnp.sqrt(M*nu*beta)]])
        positions = positions + f@positions*stepSize[:,None,None] + G@(jnp.sqrt(stepSize[:,None,None])*random.normal(key, shape=positions.shape))
        return positions.reshape((experimental_batch_size,2,2))

    def get_theoretical_covariance( i ) :
        time_indices = i
        time = util.timeIndices2RealTime( time_indices , num_timesteps)
        
        return jnp.array([[Sigma_xx(time),Sigma_xv(time)],[Sigma_xv(time) , Sigma_vv(time)]])

    def get_experimental_covariance(positions) :
        """ 
        positions : shape (experimental_batch_size,2,dim)
        """ 
        positions = positions.transpose((0,2,1))
        positions = positions - jnp.mean(positions, axis = 0)[None,...]
        
        return jnp.mean( positions[...,None]@positions[...,None,:] , axis = 0 )

    def get_theoretical_average_pos( i , init_position) :
        time_indices = i
        time = jnp.array(util.timeIndices2RealTime( time_indices, num_timesteps ))
        mu_result = mu_global_HSM(time, init_position, v_0_batch = None)

        return jnp.array( mu_result[0,:,:] )

    def get_experimental_average_pos(positions) :
        """ 
        positions : shape (experimental_batch_size,2,dim)
        """ 
        return( jnp.mean(positions, axis = 0))

    def forward_covariances(init_position, key) :
        """ 
        intput :
        - init_position : shape (2)
        """ 

        print(util.green(" calculating the theoretical and empirical covariances and averages of diffusion process, \
                          results are saved in result directory and should be verified by human expert ...") )
        
        init_position = init_position.reshape(1,2)
        key, subkey1, subkey2 = random.split(key,3)
        experimental_covariance_array = jnp.zeros(shape= (num_timesteps, 2, 2, 2))
        theoretical_covariance_array = jnp.zeros(shape= (num_timesteps,2, 2))
        theoretical_average_pos_array = jnp.zeros(shape= (num_timesteps,2, 2))
        experimental_average_pos_array = jnp.zeros(shape= (num_timesteps,2, 2))
        positions = jnp.zeros(shape = (num_timesteps, experimental_batch_size, 2,2) )
        positions = positions.at[0,:,0,:].set(init_position)

        # noise the input according to Sigma_0
        positions = positions.at[0,:,0,:].set(positions[0,:,0,:] + jnp.sqrt(Sigma_xx_0)*random.normal(subkey1, shape= positions[0,:,0,:].shape))
        positions = positions.at[0,:,1,:].set(positions[0,:,1,:] + jnp.sqrt(Sigma_vv_0)*random.normal(subkey2, shape= positions[0,:,1,:].shape))

        stepSize = util.timeSteps2stepSize(num_timesteps, experimental_batch_size)

        for i in range(num_timesteps) :
            theoretical_covariance_array = theoretical_covariance_array.at[i].set(get_theoretical_covariance(i))
            experimental_covariance_array = experimental_covariance_array.at[i].set(get_experimental_covariance(positions[i]))
            theoretical_average_pos_array = theoretical_average_pos_array.at[i].set(get_theoretical_average_pos(i, init_position))
            experimental_average_pos_array = experimental_average_pos_array.at[i].set(get_experimental_average_pos(positions[i]))

            positions = positions.at[i+1].set( forward_step(positions[i],stepSize[i], key))
            key, = random.split(key,1)

        plot_covariances_evolution(theoretical_covariance_array, experimental_covariance_array ,theoretical_average_pos_array,experimental_average_pos_array)

    def plot_covariances_evolution(theoretical_covariance_array, experimental_covariance_array ,theoretical_average_pos_array, experimental_average_pos_array) :
        timesteps = list(range(num_timesteps))

        fig, axs = plt.subplots(2, 2)

        for k in range(2):
            for j in range(2):
                axs[k,j].plot(timesteps, experimental_covariance_array[:,0,k,j], label='experimental1')
                axs[k,j].plot(timesteps, experimental_covariance_array[:,1,k,j], label='experimental2')
                axs[k,j].plot(timesteps, theoretical_covariance_array[:,k,j], label='theoretical')
                axs[k,j].legend()
        fig.savefig( os.path.join(opt.saving_folder, 'Sigma_curves.png'))
        
        fig, axs = plt.subplots(2)
        for k in range(2) :
            axs[k].plot(timesteps, experimental_average_pos_array[:,k ,0],  label='experimental1')
            axs[k].plot(timesteps, experimental_average_pos_array[:,k ,1],  label='experimental2')
            axs[k].plot(timesteps, theoretical_average_pos_array[:,k ,0],  label='theoretical1')
            axs[k].plot(timesteps, theoretical_average_pos_array[:,k ,1],  label='theoretical1')
            axs[k].legend()
        fig.savefig( os.path.join(opt.saving_folder, 'mu_curves.png'))

        return
    
    forward_covariances(jnp.array([1,1]), random.PRNGKey(7))




def plot_score_field(opt, order_plotted = 2):
    """ 
    input :
    - order_plotted : int : the order of which the score vector field is plotted, 1 is for positions, 2 is for velocities
    """ 
    batch_size = opt.train_batch_size
    num_timesteps = opt.num_timesteps
    score = opt.score
    parameters = opt.parameters

    index_order =  order_plotted - 1
    # test 
    batch_velocities = jnp.zeros((batch_size, 2, 1))
    # Batch_size must be superior to 400 for this test
    for timestep in range(1,num_timesteps,100) :
        x,y = np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20))
        X = x.flatten()
        Y = y.flatten()
        batch = np.zeros((batch_size,2,1))
        batch[:400] = np.concatenate( [X[:,None],Y[:,None]], axis = 1)[...,None]
        time_indices = np.ones((batch_size,))*timestep
        scores = score(parameters, batch, batch_velocities, time_indices)[:400]
        u,v = scores[:,index_order,0], scores[:,index_order,1]

        # print( scores[:,0,0])
        plt.figure(figsize=(4,3))
        plt.quiver(x,y,u,v)
        plt.title( "timestep : " + str(timestep))
        plt.show()


def score_norm_estimate(opt) :

    # NOTE is based on the same architecture as generate_batch in runner.py

    print(util.green("calculating score norm ..."))

    key1, opt.key = random.split(opt.key)
    num_timesteps = opt.num_timesteps 
    batch_size = 100000 # try this here
    M = opt.M 
    parameters = opt.parameters 
    Gamma = opt.Gamma
    beta = opt.beta
    nu = opt.nu
    score = opt.score

    # NOTE the score norm is ponderated by respectively 
    score_ponderated_norm_integral = np.zeros((batch_size,2)) # shape (batch_size, order)

    # NOTE it is a bit suboptimal to recompile it every time . But for simple tests it might do. Compilation time is negligeable to us.

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
        score_norm = jnp.concatenate( ( Gamma*jnp.sqrt( jnp.sum( score_x**2, axis = (1,2) ))[:,None] , M*nu*jnp.sqrt(jnp.sum( score_v**2, axis = (1,2) ))[:,None]  ), axis = 1 )*beta*step_size[:,None]
        return( batch , score_norm )
    
    time_indices = jnp.array( list(range(num_timesteps))*batch_size).reshape(batch_size,num_timesteps).T
    stepSize = util.timeSteps2stepSize(num_timesteps, batch_size)
    batch = random.normal(key1, shape = (batch_size, 2, 2,1))*jnp.array([1,M]).reshape((1,-1,1,1)) # prior distribution

    for i in tqdm(range(num_timesteps - 1 , 0, -1)) :

        batch, score_norm = predictor( batch, i, stepSize[i], parameters, opt.key )
        score_ponderated_norm_integral += np.array(score_norm)
        variance_estimate = np.var( np.mean( score_ponderated_norm_integral.reshape( batch_size//50, 50, 2 ) , axis = 0), axis = 0)
        tot_score_ponderated_norm_integral = np.mean( score_ponderated_norm_integral, axis = 0)
    print(util.blue("The estimated score norm integral is {}, with an estimated variance on the measure bounded by {}".format(tot_score_ponderated_norm_integral,variance_estimate)))
    
    opt.estimated_score_norm_integral = tot_score_ponderated_norm_integral
    opt.variance_score_norm_integral = variance_estimate

    save_opt(opt)

    return