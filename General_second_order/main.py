
import options
from runner import Runner

from jax import random
import util

def main(opt):
    run = Runner(opt)
    # if opt.task == 'train' and opt.resume_training == True:
    #     run.resume_training(opt)
    # elif opt.task == 'train' and opt.resume_training == False:
    #     run.initialize_training(opt)

    run.train_or_retrieve_model(opt)
    run.generate_deterministic_batch(opt)
    run.generate_batch( opt)
    # run.get_score_norm_integral(opt)

if __name__ == "__main__":

    print(util.yellow("======================================================="))
    print(util.yellow("     Diffusion : General parametrization"))
    print(util.yellow("======================================================="))

    print(util.magenta("setting configurations..."))
    opt = options.set()

    print(util.red("\n values    :   nu = {}    Gamma = {}    M = {} \n".format(  opt.nu, opt.Gamma, opt.M)))
    if 4*opt.M**(-1) != (opt.Gamma - opt.nu)**2 :
        print(util.red("NON-CRITICALLY DAMPED REGIME \n "))
    else :
        print(util.red("CRITICALLY DAMPED REGIME \n "))

    main(opt)
