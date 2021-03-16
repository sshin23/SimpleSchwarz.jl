module SimpleSchwarz

import Printf: @sprintf
import LightGraphs: Graph, add_edge!, neighbors, nv
import SimpleNLModels: Model, variable, parameter, constraint, objective,
    num_variables, num_constraints, func, deriv, optimize!, instantiate!
import LinearAlgebra: norm

export SchwarzModel, iterate!, optimize!

mutable struct SubModel
    model::Model
    
    x_V_orig
    l_V_orig
    x_V_sub
    l_V_sub
    
    x_bdry_orig
    l_bdry_orig
    x_bdry_sub
    l_bdry_sub

    x_err_orig
    l_err_orig
    x_err_sub
    l_err_sub
    
    function SubModel(m::Model,V,W,rho;opt...)
        msub = Model(m.optimizer;m.opt...,opt...)
        

        V_con = Int[]
        for i=1:num_constraints(m)
            minimum(keys(deriv(m.cons[i]))) in V && push!(V_con,i)
        end
        
        W_con = copy(V_con)
        W_bdry_con = Int[]

        W_bool = falses(num_variables(m))
        W_bool[W] .= true
        
        for i in setdiff(1:num_constraints(m),V_con)
            typ = examine(W_bool,keys(deriv(m.cons[i])))
            if typ == 2
                push!(W_con,i)
            elseif typ == 1
                push!(W_bdry_con,i)
            end
        end
            
        V_compl = setdiff(W,V)
        V_compl_con = setdiff(W_con,V_con)
        W_compl = setdiff(1:num_variables(m),W)

        x = [i in W ? variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i]) : parameter(msub) for i=1:num_variables(m)]
        l = Dict(i=>parameter(msub,0.) for i in W_bdry_con)
        
        for obj in m.objs
            examine(W_bool,keys(deriv(obj))) >= 1 && objective(msub,func(obj)(x,m.p))
        end

        for i in W_con
            constraint(msub,func(m.cons[i])(x,m.p);lb=m.gl[i],ub=m.gu[i])
        end
        for i in W_bdry_con
            objective(msub, (func(m.cons[i])(x,m.p)-m.gl[i]) * l[i] + 0.5 * rho * (func(m.cons[i])(x,m.p)-m.gl[i])^2 )
        end

        instantiate!(msub)

        
        x_V_orig = view(m.x,V)
        l_V_orig = view(m.l,V_con)
        x_V_sub = view(msub.x,[i for i in eachindex(W) if W[i] in V])
        l_V_sub = view(msub.l,[i for i in eachindex(W_con) if W_con[i] in V_con])

        x_bdry_orig= view(m.x,W_compl) 
        l_bdry_orig= view(m.l,W_bdry_con)
        x_bdry_sub = view(msub.p,1:length(W_compl))
        l_bdry_sub = view(msub.p,length(W_compl)+1:length(W_compl)+length(W_bdry_con))

        x_err_orig = view(m.x,V_compl)
        l_err_orig = view(m.l,V_compl_con)
        x_err_sub = view(msub.x,[i for i in eachindex(W) if W[i] in V_compl])
        l_err_sub = view(msub.l,[i for i in eachindex(W_con) if W_con[i] in V_compl_con])


        
        return new(msub,
                   x_V_orig,l_V_orig,x_V_sub,l_V_sub,
                   x_bdry_orig,l_bdry_orig,x_bdry_sub,l_bdry_sub,
                   x_err_orig,l_err_orig,x_err_sub,l_err_sub)
    end
end

mutable struct SchwarzModel
    model::Model
    submodels::Vector{SubModel}
    rho::Float64
end


function instantiate!(schwarz::SchwarzModel)
    Threads.@threads for sm in schwarz.submodels
        instantiate!(sm)
    end
end
function iterate!(schwarz::SchwarzModel;err=Threads.Atomic{Float64}(Inf))
    Threads.@threads for sm in schwarz.submodels
        set_submodel!(sm)
    end
    Threads.@threads for sm in schwarz.submodels
        optimize!(sm)
    end
    Threads.@threads for sm in schwarz.submodels
        set_orig!(sm)
    end
    Threads.@threads for sm in schwarz.submodels
        set_err!(sm,err)
    end
end
function optimize!(schwarz::SchwarzModel;tol = 1e-8, maxiter = 100,optional = schwarz->nothing)
    err=Threads.Atomic{Float64}(Inf)

    iter = 0
    while err[] > tol && iter < maxiter
        err[] = .0
        iterate!(schwarz;err=err)
        println(@sprintf "%4i %4.2e" iter+=1 err[])
        optional(schwarz)
    end
end

function set_err!(sm,err)
    Threads.atomic_max!(err,difference(sm.x_err_orig,sm.x_err_sub)/norm(sm.x_err_orig,Inf))
    Threads.atomic_max!(err,difference(sm.l_err_orig,sm.l_err_sub)/norm(sm.l_err_orig,Inf))
end

function difference(a,b)
    diff = .0
    @inbounds @simd for i in eachindex(a)
        diff = max(diff,abs(a[i] - b[i]))
    end
    return diff
end

function set_submodel!(sm)
    sm.x_bdry_sub .= sm.x_bdry_orig
    sm.l_bdry_sub .= sm.l_bdry_orig
    nothing
end

function set_orig!(sm)
    sm.x_V_orig .= sm.x_V_sub
    sm.l_V_orig .= sm.l_V_sub
    nothing
end

function examine(W_bool,keys)
    numtrue = 0
    for k in keys
        W_bool[k] && (numtrue += 1)
    end
    return numtrue == 0 ? 0 : numtrue == length(keys) ? 2 : 1
end
is_included(expr,W) = issubset(keys(deriv(expr)),W)
is_adjacent(expr,W) = !isdisjoint(keys(deriv(expr)),W)
optimize!(sm::SubModel) = optimize!(sm.model)
instantiate!(sm::SubModel) = instantiate!(sm.model)
function SchwarzModel(m::Model;rho=1.,omega=.1,opt...)

    m[:Ws]=expand(m,m[:Vs],omega)
    sms = Vector{SubModel}(undef,length(m[:Vs]))
    
    Threads.@threads for i in collect(keys(m[:Vs]))
        sms[i] = SubModel(m,m[:Vs][i],m[:Ws][i],rho;opt...)
    end
    
    SchwarzModel(m,sms,rho)
end

function expand(m,Vs,omega)
    
    g = Graph(m)
    Ws = Dict(k=>copy(V) for (k,V) in Vs)
    
    Threads.@threads for k in collect(keys(Ws))
        expand!(Ws[k],g,(1+omega)*length(Ws[k]))
        sort!(Ws[k])
    end

    return Ws
end


function expand!(V_om,g::Graph,max_size;
                 new_nbr=[])
    if isempty(new_nbr)
        new_nbr = Int[]
        for v in V_om
            append!(new_nbr,neighbors(g,v))
        end
        unique!(new_nbr)
        setdiff!(new_nbr,V_om)
    end
    
    old_nbr = V_om
    
    while (length(V_om) + length(new_nbr) < max_size) && length(V_om) < nv(g) && !isempty(new_nbr)
        append!(V_om,new_nbr)
        old_old_nbr = old_nbr
        old_nbr=new_nbr
        new_nbr = Int[]
        for v in old_nbr
            append!(new_nbr,neighbors(g,v))
        end
        unique!(new_nbr)
        setdiff!(new_nbr,old_old_nbr)
        setdiff!(new_nbr,old_nbr)
    end
    
    return new_nbr
end

function Graph(m::Model)
    g = Graph(num_variables(m))
    for con in m.cons
        for i in keys(deriv(con))
            for j in keys(deriv(con))
                i>j && add_edge!(g,i,j)
            end
        end
    end
    return g
end

end # module
