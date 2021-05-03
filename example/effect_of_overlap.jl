# quadrotor effect of overlap
dt= 0.005
K = 1:20
T = 60
N = round(Int,T/dt)
omegas = [.1,.5,1.]

err_schwarz=[]
for omega in omegas
    m = hehnandrea(SimpleSchwarz.Optimizer,N,K;dt=dt,tol=1e-4,rho=1,omega=omega,save_output=true)
    @time optimize!(m)
    push!(err_schwarz,m[:output])
end

plt = plot_err_profile(err_schwarz,omegas)
savefig(plt,"fig/err_profile.pdf")
