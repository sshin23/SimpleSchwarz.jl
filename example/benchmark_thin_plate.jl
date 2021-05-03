# benchmark - thin plate

(K,T,L,dt,dx) = (1:20,3600*4,1,5.,.1) 

N = round(Int,T/dt)
n = round(Int,L/dx)

m_ref = thinplate(Ipopt.Optimizer,n,N,K;dt=dt,dx=dx)
instantiate!(m_ref)
GC.gc(); GC.enable(false)
time_ref = @elapsed begin
    @time optimize!(m_ref)
end
GC.enable(true); GC.gc()
err_ref = KKTErrorEvaluator(m_ref)(m_ref.x,m_ref.l,m_ref.gl)

m_schwarz  = thinplate(SimpleSchwarz.Optimizer,n,N,K;dt=dt,dx=dx,rho=1,omega=1.,maxiter=400,tol=0,save_output=true)
GC.gc(); GC.enable(false)
SimpleSchwarz.instantiate!(m_schwarz)
SimpleSchwarz.optimize!(m_schwarz)
GC.enable(true); GC.gc()

m_admm = thinplate(SimpleADMM.Optimizer,n,N,K;dt=dt,dx=dx,print_level=0,rho=1,maxiter=400,tol=0,save_output=true)
GC.gc(); GC.enable(false)
instantiate!(m_admm)
optimize!(m_admm)
GC.enable(true); GC.gc()

plt = plot_benchmark(err_ref,time_ref,m_schwarz[:output],m_admm[:output])
savefig(plt,"fig/pde.pdf")
