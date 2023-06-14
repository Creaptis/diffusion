import jax.numpy as jnp
import jax.random as random
import jax 
import util 
from utils.modelization_util import get_usual_functions


def make_batch_loss(opt) :

    mu_global_HSM, Sigma_xx, Sigma_vv, Sigma_xv, BETA, L_Cholesky = get_usual_functions(opt)
    num_timesteps = opt.num_timesteps
    batch_size = opt.train_batch_size
    dimension = opt.dimension
    epsilon_model = opt.epsilon_model

    @jax.jit
    def batch_loss(parameters, batch_positions, key) :
        """
        input :
        - parameters : a dictionary with three keys 'params_times', 'params_positions', 'params_vitesses', 'params_global' for the respectives MLP
        - batch_positions : positions batch shape (batch_size, 2, 1)
        """
        key, key2, key3, key4 = random.split(opt.key,4)

        # sampling times
        time_indices = random.randint(key2, shape = (batch_size,), minval= 1, maxval= num_timesteps)
        real_time_batch = util.timeIndices2RealTime(time_indices, num_timesteps )

        epsilon = random.normal(key3, shape = (batch_size,dimension, 2)) 

        batch_global = mu_global_HSM(real_time_batch, batch_positions) + (L_Cholesky( real_time_batch )[:,None,:,:]@epsilon[...,None])[:,:,:,0].transpose( (0,2,1) ) # shape (batch_size, 2,dim)
        batch_positions = batch_global[:,0,:]
        batch_velocities = batch_global[:,1,:]


        loss = jnp.mean( (epsilon.transpose((0,2,1)) - epsilon_model.apply(parameters,batch_positions, batch_velocities, time_indices))**2 )
        return(loss)
    
    return(batch_loss)