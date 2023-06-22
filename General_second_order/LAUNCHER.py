import os
import time


def do_these_experiments() :
    set_Gamma = [0.0, 0.01, 0.02, 0.03, 0.04 ]
    set_M = [1/2.,1/3.,1/4.,1/5.,1/6.]
    set_nu = [2.82842712, 3.47410162, 4.02      , 4.50213595, 4.93897949]
    for M in set_M :
        for nu in set_nu :
            for Gamma in set_Gamma :
                time.sleep(1)
                os.system("python main.py --config-name spiral --M {M} --Gamma {Gamma} --nu {nu}".format(M=M, nu=nu, Gamma=Gamma))

if __name__ == "__main__" :
    do_these_experiments()



