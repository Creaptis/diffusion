import numpy as np

import os
import json
import util
import pickle
import optax

def save_opt(opt) :
    """
    NOTE : ensure that opt.index_iter has been set to the current index of iteration before saving

    """
    # NOTE preprocessor allows to process opt before saving to remove all elements not of basic types
    preprocessor = util.SkipFilter([int, str, float, bool, list], ["parameters"])

    # TODO  tranform to lists
        # opt.estimated_score_norm_integral = tot_score_ponderated_norm_integral
    # opt.variance_score_norm_integral = variance_estimate

    savingParamsFile = os.path.join(opt.saving_folder , "parameters.npy")
    savingOptimStateFile = os.path.join(opt.saving_folder , "optimizer_state.npy")

    print(util.green("Saving progression ..." ))
    

    with open(savingParamsFile, 'wb') as file:
        pickle.dump(opt.parameters, file)
    
    with open(savingOptimStateFile, 'wb') as file:
        pickle.dump(opt.optimizer_state , file)

    with open( os.path.join(opt.saving_folder, 'opt.txt'), 'w') as f:

        # NOTE important to save current key for reproductability, has to be saved as list however
        opt.key = np.array(opt.key).astype('uint32').tolist()
        opt.estimated_score_norm_integral =  np.array(opt.estimated_score_norm_integral).tolist()
        opt.variance_score_norm_integral = np.array(opt.variance_score_norm_integral).tolist()
        
        print("!!!!", preprocessor.filter(opt.__dict__))
        json.dump( preprocessor.filter(opt.__dict__), f, indent=2)

        opt.key = np.array(opt.key).astype('uint32')
        opt.estimated_score_norm_integral =  np.array(opt.estimated_score_norm_integral)
        opt.variance_score_norm_integral = np.array(opt.variance_score_norm_integral)
    
    print(util.green("Saving DONE \n" ))

    return

def load_opt(opt) :
    """ 
    NOTE : only completes an already existing opt (overwriting parameters with previously saved ones). Doesn't create opt
    """

    savingParamsFile = os.path.join(opt.saving_folder , "parameters.npy")
    savingOptimStateFile = os.path.join(opt.saving_folder , "optimizer_state.npy")

    with open(os.path.join(opt.saving_folder, 'opt.txt'), 'r') as f:
        print(util.red(" CAUTION ! : previously saved model params are being loaded, there is no verification that model parameters are the same as those of the command line "))
        opt.__dict__.update(json.load(f))

        # NOTE jax PRNK keys are essentially uint32 arrays
        opt.key = np.array(opt.key).astype('uint32')

    with open(savingParamsFile, 'rb') as file:
        opt.parameters = pickle.load(file)

    opt.optimizer = optax.adam(opt.lr)
    with open( savingOptimStateFile, 'rb') as file:
        opt.optimizer_state = pickle.load(file)
    assert hasattr(opt , "index_iter")

    return






