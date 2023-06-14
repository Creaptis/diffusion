import jax.numpy as jnp
import jax.random as random
from modelization_util import get_usual_functions
import matplotlib.pyplot as plt
import util

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

    mu_global_HSM, Sigma_xx, Sigma_vv, Sigma_xv, BETA, L_Cholesky = get_usual_functions(opt)

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
        mu_result = mu_global_HSM(time, init_position, v_0_batch = None, batch_size=1)

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
        key, subkey1, subkey2 = random.split(key,3)
        experimental_covariance_array = jnp.zeros(shape= (num_timesteps, 2, 2, 2))
        theoretical_covariance_array = jnp.zeros(shape= (num_timesteps,2, 2))
        theoretical_average_pos_array = jnp.zeros(shape= (num_timesteps,2, 2))
        experimental_average_pos_array = jnp.zeros(shape= (num_timesteps,2, 2))
        positions = jnp.zeros(shape = (num_timesteps, experimental_batch_size, 2,2) )
        positions = positions.at[0,:,0,:].set(init_position[None,:])

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

        for k in range(2):
            for j in range(2):
                plt.figure(figsize=(6,6))
                plt.plot(timesteps, experimental_covariance_array[:,0,k,j], label='experimental1')
                plt.plot(timesteps, experimental_covariance_array[:,1,k,j], label='experimental2')
                plt.plot(timesteps, theoretical_covariance_array[:,k,j], label='theoretical')
                plt.legend()
                plt.show()
            
        for k in range(2) :
            plt.figure(figsize=(6,6))
            plt.plot(timesteps, experimental_average_pos_array[:,k ,0],  label='experimental1')
            plt.plot(timesteps, experimental_average_pos_array[:,k ,1],  label='experimental2')
            plt.plot(timesteps, theoretical_average_pos_array[:,k ,0],  label='theoretical1')
            plt.plot(timesteps, theoretical_average_pos_array[:,k ,1],  label='theoretical1')
            plt.legend()
            plt.show()

        return
    
    forward_covariances(jnp.array([1,1]), random.PRNGKey(7))
