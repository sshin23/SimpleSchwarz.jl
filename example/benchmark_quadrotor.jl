# benchmark - quadrotor

(dt,K,T) = (0.005,1:20,60) 

N = round(Int,T/dt)

m_ref = hehnandrea(Ipopt.Optimizer,N,K;dt=dt)
instantiate!(m_ref)
GC.gc(); GC.enable(false)
time_ref = @elapsed begin
    @time optimize!(m_ref)
end
GC.enable(true); GC.gc()
set_KKT_error_evaluator!(m_ref)
err_ref = get_KKT_error(m_ref)

# m_schwarz  = hehnandrea(SimpleSchwarz.Optimizer,N,K;dt=dt,rho=1,omega=1.,maxiter=400,tol=0,save_output=true)
# GC.gc(); GC.enable(false)
# SimpleSchwarz.instantiate!(m_schwarz)
# SimpleSchwarz.optimize!(m_schwarz)
# GC.enable(true); GC.gc()

# m_admm = hehnandrea(SimpleADMM.Optimizer,N,K;dt=dt,print_level=0,rho=1,maxiter=400,tol=0,save_output=true)
# GC.gc(); GC.enable(false)
# instantiate!(m_admm)
# optimize!(m_admm)
# GC.enable(true); GC.gc()

# plt = plot_benchmark(err_ref,time_ref,m_schwarz[:output],m_admm[:output])
# savefig(plt,"fig/ocp.pdf")

