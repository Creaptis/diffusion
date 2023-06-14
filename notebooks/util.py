import jax.numpy as jnp
import jax 
import termcolor


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])



def timeSteps2stepSize(num_timesteps : int, batch_size : int):
    """ 
    - num_timesteps : int 
    - batch_size : int

    returns :
    - stepSize : shape (num_timesteps, batch_size) 
    """ 
    stepSize = jnp.ones((batch_size,num_timesteps)).T*1.0/num_timesteps
    return(stepSize)

@jax.jit
def PositionalEncoding(time_indices, opt) :
    """
    - time_indices : shape (batch_size,) , should be int between 0 and num_timesteps
    - embed_dimension : dimension of the embeded time, *must be even* for definition of k in the function !
    """
    embed_dimension = opt.mlp_intermediate_embeding_time_dimension
    
    magic_number = 10000 # NOTE custom magic number addapted for embed_dimension = 128, max_time = 2000 might need to change number for bigger max_time
    times = time_indices.reshape( (1,-1) )
    k = jnp.arange( 0, embed_dimension//2 ).reshape((-1,1))

    embedding = jnp.array([ jnp.cos( times/(magic_number**(2.*k/embed_dimension)) ) , jnp.sin( times/(magic_number**(2.*k/embed_dimension)) ) ])
    # embedding shape is (2, embed_dimension/2, batch_size) and should become (batch_size, embed_dim)
    embedding = jnp.transpose(embedding, (2,1,0))
    embedding = embedding.reshape((-1, embed_dimension ))
    ###

    return(embedding)