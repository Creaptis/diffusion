import jax.numpy as jnp
import util

# TODO implement the case where -4 + M*(Gamma-nu)**2 < 0
def initialize_usual_functions(opt) :
    beta = opt.beta
    Gamma = opt.Gamma
    nu = opt.nu
    M = opt.M
    Sigma_xx_0 = opt.Sigma_xx_0
    Sigma_vv_0 = opt.Sigma_vv_0
    dimension = opt.dimension
    batch_size = opt.train_batch_size
    tiny_tiny_value = 10**(-7)

    BETA = lambda t : t*beta

    # check if we are in the critically damped regime
    if 4*M**(-1) < (Gamma - nu)**2 - tiny_tiny_value :
        Sigma_xx = lambda t : (1/(2*M*(-4 + M*(Gamma-nu)**2)))*jnp.exp(-((BETA(t)*(jnp.sqrt(M*(-4 + M*(Gamma-nu)**2)) + M*(Gamma+nu)))/(2*M)))*(2*jnp.exp((BETA(t)*(jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)) + M*(Gamma+ nu)))/(2*M))*M*(-4 + M*(Gamma- nu)**2) + 2*Sigma_vv_0 - 4*jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/(2*M))*(Sigma_vv_0 + M*(-2 + Sigma_xx_0)) + M*(jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)) + M*(Gamma- nu))*(Gamma- nu)*(-1 + Sigma_xx_0) - 2*M*Sigma_xx_0 + jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/ M)*(2*Sigma_vv_0 - M*(Gamma- nu)*(jnp.sqrt( M*(-4 + M*(Gamma- nu)**2)) + M*(-Gamma+ nu))*(-1 + Sigma_xx_0) - 2*M*Sigma_xx_0))
        Sigma_xv = lambda t : (1/(2*M*(-4 + M*(Gamma-nu)**2)))*jnp.exp(-((BETA(t)*(jnp.sqrt(M*(-4 + M*(Gamma-nu)**2)) + M*(Gamma+nu)))/(2*M)))*(-jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*Sigma_vv_0 + jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/M)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*Sigma_vv_0 + M*(Gamma- nu)*(Sigma_vv_0 + M*(-2 + Sigma_xx_0)) - 2*jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/(2*M))*M*(Gamma- nu)*(Sigma_vv_0 + M*(-2 + Sigma_xx_0)) + jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/M)*M*(Gamma- nu)*(Sigma_vv_0 + M*(-2 + Sigma_xx_0)) + M*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*Sigma_xx_0 -jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/M)*M*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*Sigma_xx_0)
        Sigma_vv = lambda t : -(1/(2*(-4 + M*(Gamma-nu)**2)))*jnp.exp(-((BETA(t)*(jnp.sqrt(M*(-4 + M*(Gamma-nu)**2)) + M*(Gamma+nu)))/(2*M)))*(-2*jnp.exp((BETA(t)*(jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)) + M*(Gamma+ nu)))/(2*M))*M*(-4 + M*(Gamma- nu)**2) + M**2*(Gamma- nu)**2 + (2 + jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*(Gamma- nu))*Sigma_vv_0 + 4*jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/(2*M))*(Sigma_vv_0 + M*(-2 + Sigma_xx_0)) - M*((Gamma- nu)*(jnp.sqrt( M*(-4 + M*(Gamma- nu)**2)) + Gamma*Sigma_vv_0 - nu*Sigma_vv_0) + 2*Sigma_xx_0) + jnp.exp((BETA(t)*jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)))/M)*(M**2*(Gamma- nu)**2 + 2*Sigma_vv_0 + jnp.sqrt(M*(-4 + M*(Gamma- nu)**2))*(-Gamma+ nu)*Sigma_vv_0 + M*(Gamma- nu)*(jnp.sqrt(M*(-4 + M*(Gamma- nu)**2)) - Gamma*Sigma_vv_0 + nu*Sigma_vv_0) - 2*M*Sigma_xx_0))

        def mu_global_HSM(t_batch, x_0_batch, v_0_batch = None ) :
            """ 
            input :
            - t_batch : (batch_size,)
            - x_0_batch : (batch_size,dimension,1) or (batch_size,dimension)
            - v_0_batch (optional) : (batch_size,dimension,1) or (batch_size,dimension)
            output :
            - mu : (batch_size,2,dimension)
            """

            assert len(x_0_batch.shape) >=2 , "in case of batch_size = 1, x_0 should still have a batch dimension"
            batch_size = x_0_batch.shape[0]

            assert len(x_0_batch.shape) >=2 , "in case of batch_size = 1, x_0 should still have a batch dimension"
            batch_size = x_0_batch.shape[0]

            if v_0_batch is None :
                v_0_batch = jnp.zeros(x_0_batch.shape)
            
            x0 = x_0_batch.reshape(batch_size, dimension)
            v0 = v_0_batch.reshape(batch_size, dimension)
            t = t_batch.reshape((-1,1))

            mu_v = (jnp.exp(-((BETA(t)*(jnp.sqrt(-4 + M*(Gamma - nu)**2) + jnp.sqrt(M)*(Gamma + nu)))/( 4*jnp.sqrt(M))))*((1 + jnp.exp(( BETA(t)*jnp.sqrt(-4 + M*(Gamma - nu)**2))/(2*jnp.sqrt(M))))*v0*jnp.sqrt(-4 + M*(Gamma - nu)**2) - (-1 + jnp.exp(( BETA(t)*jnp.sqrt(-4 + M*(Gamma - nu)**2))/( 2*jnp.sqrt(M))))*jnp.sqrt(M)*(2*x0 + v0*(-Gamma + nu))))/( 2*jnp.sqrt(-4 + M*(Gamma - nu)**2) )
            mu_x = (jnp.exp(-((BETA(t)*(jnp.sqrt(-4 + M*(Gamma - nu)**2) + jnp.sqrt(M)*(Gamma + nu)))/( 4*jnp.sqrt(M))))*(2*(-1 + jnp.exp(( BETA(t)*jnp.sqrt(-4 + M*(Gamma - nu)**2))/( 2*jnp.sqrt(M))))*v0 + (1 + jnp.exp(( BETA(t)*jnp.sqrt(-4 + M*(Gamma - nu)**2))/(2*jnp.sqrt(M))))*jnp.sqrt(M)*x0*jnp.sqrt(-4 + M*(Gamma - nu)**2) - (-1 + jnp.exp(( BETA(t)*jnp.sqrt(-4 + M*(Gamma - nu)**2))/( 2*jnp.sqrt(M))))*M*x0*(Gamma - nu)))/( 2*jnp.sqrt(M)*jnp.sqrt(-4 + M*(Gamma - nu)**2))
            
            # (2, batch_size ,  dim) -> (batch_size, 2, dim)
            return jnp.array([mu_x, mu_v ]).transpose( (1,0,2) )
    elif 4*M**(-1) > (Gamma - nu)**2 + tiny_tiny_value :
        Sigma_xx = lambda t : (-4*Sigma_vv_0*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M)))**2)/(jnp.exp((BETA(t)*(Gamma + nu))/2)*M*(-4 + M*(Gamma - nu)**2)) + (Sigma_xx_0*(-2 + (-2 + M*(Gamma - nu)**2)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) + jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*(Gamma - nu)*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M))))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2)) + (4 + jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2) - M*(Gamma - nu)**2*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) + jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*(-Gamma + nu)*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M)))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2))
        Sigma_xv = lambda t : (4*M*(Gamma - nu)*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M)))**2)/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2)) + (Sigma_vv_0*(M*(-Gamma + nu) + M*(Gamma - nu)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) - jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M))))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*M*(-4 + M*(Gamma - nu)**2)) + (Sigma_xx_0*(M*(-Gamma + nu) + M*(Gamma - nu)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) + jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M))))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2))  
        Sigma_vv = lambda t : (-4*M*Sigma_xx_0*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M)))**2)/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2)) + (M*(4 + jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2) - M*(Gamma - nu)**2*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) + jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*(Gamma - nu)*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M))))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2)) + (Sigma_vv_0*(-2 + (-2 + M*(Gamma - nu)**2)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(2*jnp.sqrt(M))) + jnp.sqrt(M*(4 - M*(Gamma - nu)**2))*(-Gamma + nu)*jnp.sin((BETA(t)*jnp.sqrt(M*(4 - M*(Gamma - nu)**2)))/(2*M))))/(jnp.exp((BETA(t)*(Gamma + nu))/2)*(-4 + M*(Gamma - nu)**2))

        def mu_global_HSM(t_batch, x_0_batch, v_0_batch = None, batch_size = batch_size) :
            """ 
            input :
            - t_batch : (batch_size,)
            - x_0_batch : (batch_size,dimension,1) or (batch_size,dimension)
            - v_0_batch (optional) : (batch_size,dimension,1) or (batch_size,dimension)
            output :
            - mu : (batch_size,2,dimension)
            """
            assert len(x_0_batch.shape) >=2 , "in case of batch_size = 1, x_0 should still have a batch dimension"
            batch_size = x_0_batch.shape[0]

            if v_0_batch is None :
                v_0_batch = jnp.zeros(x_0_batch.shape)
                
            x0 = x_0_batch.reshape(batch_size, dimension)
            v0 = v_0_batch.reshape(batch_size, dimension)
            t = t_batch.reshape((-1,1))

            mu_v = (-2*jnp.sqrt(M)*x0*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M))))/(jnp.exp((BETA(t)*(Gamma + nu))/4)*jnp.sqrt(4 - M*(Gamma - nu)**2)) + (v0*(jnp.sqrt(4 - M*(Gamma - nu)**2)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M))) + jnp.sqrt(M)*(Gamma - nu)*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M)))))/(jnp.exp((BETA(t)*(Gamma + nu))/4)*jnp.sqrt(4 - M*(Gamma - nu)**2))
            
            mu_x = (2*v0*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M))))/(jnp.exp((BETA(t)*(Gamma + nu))/4)*jnp.sqrt(M)*jnp.sqrt(4 - M*(Gamma - nu)**2)) + (x0*(jnp.sqrt(4 - M*(Gamma - nu)**2)*jnp.cos((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M))) + jnp.sqrt(M)*(-Gamma + nu)*jnp.sin((BETA(t)*jnp.sqrt(4 - M*(Gamma - nu)**2))/(4*jnp.sqrt(M)))))/(jnp.exp((BETA(t)*(Gamma + nu))/4)*jnp.sqrt(4 - M*(Gamma - nu)**2))
            
            # (2, batch_size ,  dim) -> (batch_size, 2, dim)
            return jnp.array([mu_x, mu_v ]).transpose( (1,0,2) )
    else :
        A1 = 1./(4*M)
        A2 = M**(-2)/8.
        A2 = M**(-2)/4. # TODO REVOIR !!
        A3 = (nu-Gamma)/2.
        A4 = -M**(-1)/2. 
        A5 = (Gamma-nu)/2.
        C1 = (Gamma-nu)/8.
        C2 = (Gamma-nu)**3/32.
        C3 = -1/2.
        C4 = M**(-1)/2.
        C5 = (nu-Gamma)/4.
        D1 = 1/4.
        D2 = M**(-1)/4.
        D3 = (Gamma-nu)/2.
        D4 = -1/2.
        D5 = M*(nu-Gamma)/4.
        Sigma_xx = lambda t : jnp.exp(-(Gamma+nu)/2.*BETA(t))*( A1*BETA(t)**2*Sigma_xx_0 + A2*BETA(t)**2*Sigma_vv_0 + A3*BETA(t)*Sigma_xx_0 + A4*BETA(t)**2 + A5*BETA(t) + (jnp.exp((Gamma + nu)/2.*BETA(t)) - 1) + Sigma_xx_0)
        Sigma_xv = lambda t : jnp.exp(-(Gamma+nu)/2.*BETA(t))*( C1*BETA(t)**2*Sigma_xx_0 + C2*BETA(t)**2*Sigma_vv_0 + C3*BETA(t)*Sigma_xx_0 + C4*BETA(t)*Sigma_vv_0 + C5*BETA(t)**2)
        Sigma_vv = lambda t : jnp.exp(-(Gamma+nu)/2.*BETA(t))*( D1*BETA(t)**2*Sigma_xx_0 + D2*BETA(t)**2*Sigma_vv_0 + D3*BETA(t)*Sigma_vv_0 + D4*BETA(t)**2 + D5*BETA(t) + M*(jnp.exp((Gamma + nu)/2.*BETA(t)) - 1) + Sigma_vv_0)

        Sigma_xx = lambda t : 1+1/8*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*(-8-BETA(t)*(-4+BETA(t)*(Gamma-nu))*(Gamma-nu))+1/64*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*BETA(t)**2*(Gamma-nu)**4*Sigma_vv_0+1/16*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*(-4+BETA(t)*(Gamma-nu))**2* Sigma_xx_0
        Sigma_xv = lambda t : 1/4*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*BETA(t)**2*(-Gamma+nu)+1/32*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*BETA(t)*(4+BETA(t)*(Gamma-nu))*(Gamma-nu)**2* Sigma_vv_0+1/8*jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*BETA(t)*(-4+BETA(t)*(Gamma-nu))*Sigma_xx_0
        Sigma_vv = lambda t : (jnp.exp(-(1/2)*BETA(t)*(Gamma+nu))*(-8+8*jnp.exp(1/2*BETA(t)*(Gamma+nu))-BETA(t)* (4+BETA(t)*(Gamma-nu))*(Gamma-nu)))/(2*(Gamma-nu)**2)+1/16*jnp.exp(-(1/2)*BETA(t)* (Gamma+nu))*(4+BETA(t)*(Gamma-nu))**2*Sigma_vv_0+1/4*jnp.exp(-(1/2)*BETA(t)* (Gamma+nu))*BETA(t)**2*Sigma_xx_0

        def mu_global_HSM(t_batch, x_0_batch, v_0_batch = None) :
            """ 
            input :
            - t_batch : (batch_size,)
            - x_0_batch : (batch_size,dimension,1) or (batch_size,dimension)
            - v_0_batch (optional) : (batch_size,dimension,1) or (batch_size,dimension)
            output :
            - mu : (batch_size,2,dimension)
            """
            assert len(x_0_batch.shape) >=2 , "in case of batch_size = 1, x_0 should still have a batch dimension"
            batch_size = x_0_batch.shape[0]
            if v_0_batch is None :
                v_0_batch = jnp.zeros(x_0_batch.shape)
            
            x0 = x_0_batch.reshape(batch_size, dimension)
            v0 = v_0_batch.reshape(batch_size, dimension)
            t = t_batch.reshape((-1,1))

            mu_v =  jnp.exp(-(BETA(t)*(Gamma + nu)/4.0))*(  -1/2.*BETA(t)*x0 + (Gamma-nu)/4.*BETA(t)*v0 + v0 )
            mu_x = jnp.exp(-(BETA(t)*(Gamma + nu)/4.0))*(  (nu-Gamma)/4.*BETA(t)*x0 + (nu-Gamma)**2/8.*BETA(t)*v0 + x0  )

            
            # (2, batch_size ,  dim) -> (batch_size, 2, dim)
            return jnp.array([mu_x, mu_v ]).transpose( (1,0,2) )
        
    L_Cholesky = lambda t : jnp.array([ [jnp.sqrt(Sigma_xx(t)) , jnp.zeros( t.shape ) ] ,\
                                  [ Sigma_xv(t)/jnp.sqrt(Sigma_xx(t)) , jnp.sqrt( (Sigma_xx(t)*Sigma_vv(t) - Sigma_xv(t)**2)/Sigma_xx(t) ) ]]).transpose((2,0,1))
    
    def L_Cholesky_inv_transpose(t_) :
        return jnp.array([ 
            [ 1./jnp.sqrt(Sigma_xx(t_)) ,  -Sigma_xv(t_)/jnp.sqrt((Sigma_vv(t_)*Sigma_xx(t_) - Sigma_xv(t_)**2)*Sigma_xx(t_))  ] ,
            [ jnp.zeros( t_.shape )     ,  jnp.sqrt(Sigma_xx(t_)/(Sigma_vv(t_)*Sigma_xx(t_) - Sigma_xv(t_)**2))                ]
            ]).transpose((2,0,1)) 
    
    opt.mu_global_HSM  = mu_global_HSM 
    opt.Sigma_xx = Sigma_xx
    opt.Sigma_vv = Sigma_vv
    opt.Sigma_xv = Sigma_xv
    opt.BETA = BETA
    opt.L_Cholesky = L_Cholesky
    opt.L_Cholesky_inv_transpose = L_Cholesky_inv_transpose 
    
    return


def get_score(opt) :
    L_Cholesky_inv_transpose = opt.L_Cholesky_inv_transpose
    epsilon_model = opt.epsilon_model
    num_timesteps = opt.num_timesteps

    def score(parameters,batch_positions, batch_velocities, time_indices) :
        """ 
        input :
        - TODO write
        output :
        - shape (batch_size,2,dim)
        """
        real_time_batch = util.timeIndices2RealTime(time_indices , num_timesteps)
        return( -(L_Cholesky_inv_transpose(real_time_batch)[:,None,:,:]@epsilon_model.apply(parameters,batch_positions, batch_velocities, time_indices).transpose((0,2,1))[:,:,:,None])[:,:,:,0].transpose((0,2,1)) )
    return(score)