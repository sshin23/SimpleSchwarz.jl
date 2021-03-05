module SchwarzNLP

import SimpleNLModels: Model, variable, parameter, constraint, objective,
    num_variables, num_constraints, func, deriv, optimize!, instantiate!

export SubModel, SchwarzModel, CompositeVector, iterate!

mutable struct SubModel
    model::Model
    W::AbstractVector{Int}
    V::AbstractVector{Int}
    V_con
    
    x_V_orig
    l_V_orig
    x_V_sub
    l_V_sub
    
    x_bdry_orig
    l_bdry_orig
    x_bdry_sub
    l_bdry_sub
    
    function SubModel(m::Model,W,V,V_con,rho;opt...)
        
        msub = Model(m.optimizer;m.opt...,opt...)

        W_con = Int[]
        W_bdry_con = Int[]
        for i=1:num_constraints(m)
            if is_included(m.cons[i],W)
                push!(W_con,i)
            elseif is_adjacent(m.cons[i],W)
                push!(W_bdry_con,i)
            end
        end

        W_compl = setdiff(1:num_constraints(m),W)

        x = [i in W ? variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i]) : parameter(msub) for i=1:num_variables(m)]
        l = Dict(i=>parameter(msub,0.) for i in W_bdry_con)
        
        for obj in m.objs
            is_adjacent(obj,W) && objective(msub,func(obj)(x))
        end

        for i in W_con
            con = m.cons[i]
            constraint(msub,func(m.cons[i])(x);lb=m.gl[i],ub=m.gu[i])
        end
        for i in W_bdry_con
            expr = func(m.cons[i])(x)-m.gl[i]
            objective(msub, + expr * l[i] + 0.5 * rho * copy(expr)^2 )
        end

        instantiate!(msub)
        
        x_V_orig = view(m.x,V)
        l_V_orig = view(m.l,V_con)
        x_V_sub = view(msub.x,[i for i in eachindex(W) if W[i] in V])
        l_V_sub = view(msub.x,[i for i in eachindex(W_con) if W_con[i] in V_con])

        x_bdry_orig= view(m.x,W_compl) 
        l_bdry_orig= view(m.l,W_bdry_con)
        x_bdry_sub = view(msub.p,1:length(W_compl))
        l_bdry_sub = view(msub.p,length(W)+1:length(W)+length(W_bdry_con))

        
        return new(msub,W,V,V_con,
                   x_V_orig,l_V_orig,x_V_sub,l_V_sub,x_bdry_orig,
                   l_bdry_orig,x_bdry_sub,l_bdry_sub)
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
function iterate!(schwarz::SchwarzModel)
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

is_included(expr,W) = issubset(keys(deriv(expr)),W)
is_adjacent(expr,W) = !isempty(intersect(keys(deriv(expr)),W))
optimize!(sm::SubModel) = optimize!(sm.model)
instantiate!(sm::SubModel) = instantiate!(sm.model)
SchwarzModel(m::Model,WVs,rho=0.;opt...) = SchwarzModel(m,[SubModel(m,W,V,V_con,rho;opt...)  for (W,V,V_con) in WVs],rho)

end
