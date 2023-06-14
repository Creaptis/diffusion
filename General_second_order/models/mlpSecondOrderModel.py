
from flax import linen as nn
import optax
import jax.numpy as jnp
import util 
import jax.random as random

intermediate_embeding_time_dimension = 128
intermediate_features_embedding = 128

class batch_MLP_bloc(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)       # shape inference
    return x


class epsilon_model_module(nn.Module) :

    @nn.compact
    def __call__(self, batch_positions, batch_velocities, time_indices):
        """
        input :
        - batch_velocities : positions batch shape (batch_size, 2, 1)
        - batch_positions : positions batch shape (batch_size, 2, 1)
        - time_indices : shape (batch_size,) should be integers (float of int)
        output :
        - shape (batch_size,2,dimension)
        """
        ### NOTE is probably useless :
        batch_velocities = batch_velocities.reshape((batch_velocities.shape[0],-1))
        batch_positions = batch_positions.reshape((batch_positions.shape[0],-1))
        ###

        time_embedding = util.PositionalEncoding(time_indices)
        time_embedding = batch_MLP_bloc(out_dims=intermediate_features_embedding , name="timeMLP")( time_embedding)
        position_embedding = batch_MLP_bloc(out_dims=intermediate_features_embedding , name="positionsMLP")( batch_positions)
        velocity_embedding = batch_MLP_bloc(out_dims=intermediate_features_embedding , name="velocityMLP")(batch_velocities)

        global_embedding = jnp.concatenate( (time_embedding, position_embedding, velocity_embedding) , axis = 1)
        result = batch_MLP_bloc( out_dims = 4 , name="GlobalMLP" )(global_embedding)[:,:,None].reshape(-1,2,2)

        return result


def generateParameters(opt):

    epsilon_model = opt.epsilon_model
    key = opt.key
    opt.key, = random.split(opt.key,1)

    batch_size = opt.train_batch_size
    
    typical_batch_positions = jnp.empty((batch_size, 2,1))
    typical_batch_velocities = jnp.empty((batch_size, 2,1))
    typical_time_indices = jnp.empty((batch_size,))
    key,subkey = random.split(key)
    parameters = epsilon_model.init(batch_positions = typical_batch_positions, batch_velocities = typical_batch_velocities, time_indices= typical_time_indices, rngs = subkey)
 
    return(parameters)
