module SimpleSchwarz

import Printf: @sprintf
import LightGraphs: Graph, add_edge!, neighbors, nv
import SimpleNL: Model, variable, parameter, constraint, objective, Expression, non_caching_eval, num_variables, num_constraints, optimize!, instantiate!
import SimpleNLUtils: get_terms, get_entries_expr, sparsity, KKTErrorEvaluator
import LinearAlgebra: norm
import Requires: @require

default_subproblem_optimizer() = @isdefined(DEFAULT_SUBPROBLEM_OPTIMIZER) ? DEFAULT_SUBPROBLEM_OPTIMIZER : error("DEFAULT_SUBPROBLEM_OPTIMIZER is not defined. To use Ipopt as a default subproblem optimizer, do: using Ipopt")

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

    function SubModel(m::Model,optimizer,V,W,rho,objs,cons,obj_sparsity,con_sparsity;opt...)
        msub = Model(optimizer;opt...)
        W_bool = falses(num_variables(m))
        W_bool[W] .= true
        V_bool = falses(num_variables(m))
        V_bool[V] .= true
        
        V_con = get_V_con(V_bool,con_sparsity)
        W_con = copy(V_con)
        W_bdry_con = Int[]

        for i in setdiff(1:num_constraints(m),V_con)
            typ = examine(W_bool,con_sparsity[i])
            if typ == 2
                push!(W_con,i)
            elseif typ == 1
                push!(W_bdry_con,i)
            end
        end
        W_obj = get_W_obj(objs,obj_sparsity,W_bool)            
        V_compl = setdiff(W,V)
        V_compl_con = setdiff(W_con,V_con)

        
        W_cl_bool = W_bool
        for sp in con_sparsity
            W_cl_bool[sp] .= true
        end
        W_cl = findall(W_cl_bool)
        W_compl = setdiff(W_cl,W)

        x = Dict(i=> W_cl_bool[i] ? variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i]) : 0. for i in W_cl)
        xc= Dict(i=> parameter(msub,0.) for i in W_compl)
        l = Dict(i=>parameter(msub,0.) for i in W_bdry_con)
        c = Dict(i=>variable(msub) for i in W_bdry_con)
        
        for i in W_con
            constraint(msub,non_caching_eval(cons[i],x,m.p);lb=m.gl[i],ub=m.gu[i])
        end
        for i in W_bdry_con
            constraint(msub, - c[i] + non_caching_eval(cons[i],x,m.p);lb=m.gl[i],ub=m.gu[i])
        end
        for i in W_compl
            constraint(msub,x[i] - xc[i])
        end
        
        objective(msub,sum(non_caching_eval(objs[i],x,m.p) for i in W_obj; init = 0) +
                  sum(c[i] * l[i] + 0.5 * rho * c[i]^2 for i in W_bdry_con; init = 0))
        
        instantiate!(msub)
        x_V_orig = view(m.x,V)
        l_V_orig = view(m.l,V_con)
        x_V_sub = view(msub.x,Base._findin(W_cl,V))
        l_V_sub = view(msub.l,Base._findin(W_con,V_con))

        x_bdry_orig= view(m.x,W_compl) 
        l_bdry_orig= view(m.l,W_bdry_con)
        x_bdry_sub = view(msub.p,1:length(W_compl))
        l_bdry_sub = view(msub.p,length(W_compl)+1:length(W_compl)+length(W_bdry_con))

        return new(msub,
                   x_V_orig,l_V_orig,x_V_sub,l_V_sub,
                   x_bdry_orig,l_bdry_orig,x_bdry_sub,l_bdry_sub)
    end
end

function get_V_con(V_bool,con_sparsity)
    V_con = Int[]
    for i=1:length(con_sparsity)
        V_bool[minimum(con_sparsity[i])] && push!(V_con,i)
    end
    return V_con
end

get_W_obj(objs,obj_sparsity,W_bool) = [i for i in eachindex(objs) if examine(W_bool,obj_sparsity[i]) >= 1]

mutable struct Optimizer
    model::Model
    submodels::Vector{SubModel}
    kkt_error_evaluator
    opt::Dict{Symbol,Any}
end

default_option() = Dict(
    :rho=>1.,
    :omega=>1.,
    :maxiter=>400,
    :tol=>1e-6,
    :subproblem_optimizer=>default_subproblem_optimizer(),
    :subproblem_option=>Dict(:print_level=>0),
    :save_output=>false,
    :optional=>schwarz->nothing
)

function instantiate!(schwarz::Optimizer)
    Threads.@threads for sm in schwarz.submodels
        instantiate!(sm)
    end
end

function optimize!(schwarz::Optimizer)
    save_output = schwarz.opt[:save_output]
    if save_output
        output = Tuple{Float64,Float64}[]
        start = time()
    end
    
    iter = 0
    while (err=schwarz.kkt_error_evaluator(schwarz.model.x,schwarz.model.l,schwarz.model.gl)) > schwarz.opt[:tol] && iter < schwarz.opt[:maxiter]
        save_output && push!(output,(err,time()-start))
        println(@sprintf "%4i %4.2e" iter+=1 err)
        
        Threads.@threads for sm in schwarz.submodels
            set_submodel!(sm)
        end
        Threads.@threads for sm in schwarz.submodels
            optimize!(sm)
        end
        Threads.@threads for sm in schwarz.submodels
            set_orig!(sm)
        end
    end
    save_output && push!(output,(err,time()-start))
    println(@sprintf "%4i %4.2e" iter+=1 err)
    save_output && (schwarz.model.ext[:output]=output)
    
    schwarz.opt[:optional](schwarz)
end

# function set_err!(sm,err)
#     Threads.atomic_max!(err,difference(sm.x_err_orig,sm.x_err_sub)/norm(sm.x_err_orig,Inf))
#     Threads.atomic_max!(err,difference(sm.l_err_orig,sm.l_err_sub)/norm(sm.l_err_orig,Inf))
# end

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

optimize!(sm::SubModel) = optimize!(sm.model)
instantiate!(sm::SubModel) = instantiate!(sm.model)

function Optimizer(m::Model)
    opt = default_option()
    for (sym,val) in m.opt
        opt[sym] = val
    end
    
    objs = get_terms(m.obj)
    cons = get_entries_expr(m.con)
    obj_sparsity = [sparsity(e) for e in objs]
    con_sparsity = [sparsity(e) for e in cons]
    
    m[:Ws]=expand(num_variables(m),con_sparsity,m[:Vs],opt[:omega])
    sms = Vector{SubModel}(undef,length(m[:Vs]))
    Threads.@threads for i in collect(keys(m[:Vs]))
        sms[i] = SubModel(m,opt[:subproblem_optimizer],m[:Vs][i],m[:Ws][i],opt[:rho],objs,cons,obj_sparsity,con_sparsity;opt[:subproblem_option]...)
    end
    Optimizer(m,sms,KKTErrorEvaluator(m),opt)
end

function expand(n,con_sparsity,Vs,omega)
    
    g = Graph(n,con_sparsity)
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

function Graph(n::Int,con_sparsity::Vector{Vector{Int}})
    g = Graph(n)
    for sp in con_sparsity
        _graph!(g,sp)
    end
    return g
end

function _graph!(g,sp)
    for i in sp
        for j in sp
            i>j && add_edge!(g,i,j)
        end
    end
end

function __init__()
    @require Ipopt="b6b21f68-93f8-5de0-b562-5493be1d77c9" @eval begin
        import ..Ipopt
        DEFAULT_SUBPROBLEM_OPTIMIZER = Ipopt.Optimizer
    end
end

end # module
