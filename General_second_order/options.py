import argparse
import configs
import os
import util 
import jax
from jax import random
from utils.distributions import distribution_dataset
from utils.distributions import NumpyLoader

import time

from utils.modelization_util import initialize_usual_functions

def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving-file-name", type=str,                           help= "name given to the file containing checkpoint and results ")
    parser.add_argument("--config-name",      type=str,                           help= "makes use of the config associated to name")
    parser.add_argument("--task",             type=str, choices=['train','test'], help= "choose train or test (checkpoint must exist)")
    parser.add_argument("--resume-training",  action="store_true",                help= "True to resume training, saving-file-name must be valid and contain checkpoint")
    parser.add_argument("--dimension",        type=int,                           help= "value of dimension of problems")
    parser.add_argument("--seed",             type=int, default=123,              help= "seed for jax.random number generation")

    # --------------- model ---------------
    parser.add_argument("--T",              type=float,  default=1.,          help="time integral end time")
    parser.add_argument("--num-timesteps",  type=int,    default=2000,        help="number of steps during inference")
    parser.add_argument("--device",         type=float ,  choices=["automatic","gpu", "cpu"] , help= "device to use, default behavior finds and uses the gpu if it exists")
    parser.add_argument("--beta",           type=float,                       help= "value of beta (unit of time) if beta is constant")
    parser.add_argument("--Gamma",          type=float,                       help= "value of Gamma")
    parser.add_argument("--M",              type=float,                       help= "value of M")
    parser.add_argument("--nu",             type=float,                       help= "value of nu")
    parser.add_argument("--Sigma_xx_0",     type=float,                       help= "value of Sigma_xx at time 0")
    parser.add_argument("--Sigma_vv_0",     type=float,                       help= "value of Sigma_vv at time 0")
    

    # --------------- optimizer and loss ---------------
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--lr",             type=float,                   help="learning rate")

    # --------------- training & sampling (corrector) ---------------
    parser.add_argument("--train-batch-size", type=int, help="batch size for sampling data during training")
    parser.add_argument("--num-train-iter",   type=str, help="number of training iteration before stopping, includes precomputed steps in case of training resuming")

    config_name = parser.parse_args().config_name
    default_config, model_configs = {
        'spiral':       configs.get_spiral_default_configs,
    }.get(config_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= auto setup & path handle =========
    if opt.device == 'automatic' :
        print( util.red("the device used is", jax.default_backend()))


    opt.key, subkey = random.split( random.PRNGKey(opt.seed))
    # initialise distribution if 
    if opt.config_name == 'spiral' :
        trainset = distribution_dataset(key=subkey , batch_size = opt.train_batch_size)
        opt.train_dataloader = NumpyLoader(trainset, batch_size=1, shuffle=True)
        opt.saving_folder = os.path.join( "results"
                                    ,"second_order" + 
                                    ("_beta" + str(opt.beta) + "_M" + str(opt.M) + "_nu" + str(opt.nu) + "_Gamma" + str(opt.Gamma)).replace(".","-"))
        
        if not os.path.isdir( opt.saving_folder ) :
            print(util.yellow( " -- creating the directory " + str(opt.saving_folder)) + " to store experimental results -- " )
            os.mkdir( opt.saving_folder )
        else :
            print(util.red("THE DIRECTORY OF THIS EXPERIEMENT ALREADY EXISTED, RECUPERATING PREVIOUS CONFIGURATIONS"))
        initialize_usual_functions(opt)
    else :
        raise RuntimeError 
        # TODO 


    opt.launch_time = time.time()


    return opt
