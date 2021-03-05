using SimpleNLModels, SchwarzNLP, MadNLP, Ipopt, LinearAlgebra

function prob(N)
    m = SimpleNLModels.Model(MadNLP.Optimizer;print_level=MadNLP.ERROR)
    # m = SimpleNLModels.Model(Ipopt.Optimizer;print_level=0)
    x = [variable(m;start=mod(i,2)==1 ? -1.2 : 1.) for i=1:N];
    objective(m,sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N))
    for i=1:N-2
        constraint(m,3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3)
    end
    return m
end

K = 4
N = 80000
M = Int(N/K)

m_ref = prob(N)
m = prob(N)

omega = 2
WVs = [(
    max(M*(i-1)+1-omega,1):min(M*(i-1)+M+omega,num_variables(m)),
    max(M*(i-1)+1,1):min(M*(i-1)+M,num_variables(m)),
    max(M*(i-1)+1,1):min(M*(i-1)+M,num_constraints(m))
) for i=1:K]

schwarz = SchwarzModel(m,WVs,100.)

instantiate!(m_ref)
# instantiate!(schwarz)

@time optimize!(m_ref)
@time for i=1:3
    iterate!(schwarz)
    println(norm(m_ref.x-schwarz.model.x,Inf))
end



nothing

