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

def timeIndices2RealTime(time_indices, num_timesteps ) :
    return( time_indices *1.0/num_timesteps)

def timeSteps2stepSize(num_timesteps : int, batch_size : int):
    """ 
    - num_timesteps : int 
    - batch_size : int

    returns :
    - stepSize : shape (num_timesteps, batch_size) 
    """ 
    stepSize = jnp.ones((batch_size,num_timesteps)).T*1.0/num_timesteps
    return(stepSize)

def PositionalEncoding(time_indices, embed_dimension = 128) :
    """
    - time_indices : shape (batch_size,) , should be int between 0 and num_timesteps
    - embed_dimension : dimension of the embeded time, *must be even* for definition of k in the function !
    """
    
    magic_number = 10000 # NOTE custom magic number addapted for embed_dimension = 128, max_time = 2000 might need to change number for bigger max_time
    times = time_indices.reshape( (1,-1) )
    k = jnp.arange( 0, embed_dimension//2 ).reshape((-1,1))

    embedding = jnp.array([ jnp.cos( times/(magic_number**(2.*k/embed_dimension)) ) , jnp.sin( times/(magic_number**(2.*k/embed_dimension)) ) ])
    # embedding shape is (2, embed_dimension/2, batch_size) and should become (batch_size, embed_dim)
    embedding = jnp.transpose(embedding, (2,1,0))
    embedding = embedding.reshape((-1, embed_dimension ))
    ###

    return(embedding)






# import collections

# useful to allow jason.dump to skip over functions 
class SkipFilter(object):

    def __init__(self, types=None, keys=None, allow_empty=False):
        # self.types = tuple(types or [])
        # self.keys = set(keys or [])
        # self.allow_empty = allow_empty  # if True include empty filtered structures
        self.allowed_types = tuple(types or [])
        self.banned_keys = set(keys or [])

    def filter(self, data):
        # if isinstance(data, collections.abc.Mapping):
        #     result = {}  # dict-like, use dict as a base
        #     for k, v in data.items():
        #         if k in self.keys or isinstance(v, self.types):  # skip key/type
        #             continue
        #         try:
        #             result[k] = self.filter(v)
        #         except ValueError:
        #             pass
        #     if result or self.allow_empty:
        #         return result
        # elif isinstance(data, collections.abc.Sequence):
        #     result = []  # a sequence, use list as a base
        #     for v in data:
        #         if isinstance(v, self.types):  # skip type
        #             continue
        #         try:
        #             result.append(self.filter(v))
        #         except ValueError:
        #             pass
        #     if result or self.allow_empty:
        #         return result
        # else:  # we don't know how to traverse this structure...
        #     return data  # return it as-is, hope for the best...

        # result = {}  # dict-like, use dict as a base
        # for k, v in data.items():
        #     if k in self.banned_keys or not isinstance(v, self.allowed_types):  # skip key/type
        #         continue
        #     try:
        #         result[k] = v
        #     except ValueError:
        #         pass

        result = {}  # dict-like, use dict as a base
        for k, v in data.items():
            if isinstance(v, self.allowed_types):
                result[k] = v
        return result
        