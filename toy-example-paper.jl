using LinearAlgebra
using ForwardDiff
using Plots
import Plots.heatmap
using Plots.PlotMeasures
using PlotlySave

### Standard functions 

# Reward of a policy
function R(π, α, β, γ, μ, r)
    τ = π * β  # Observation to state policy
    # Compute the state-state transition
    pπ = [τ[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * τ)  # Compute the one step reward
    Vπ = (I-γ*transpose(pπ))\((1-γ)*rπ) # Compute the state value function via Bellman's equation
    return μ'*Vπ
end

# State action frequency for a policy
function stateActionFrequency(π, α, β, γ, μ)
    τ = β * π
    #pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    #ρ = (I-γ*pπ)\((1-γ)*μ)
    #η = Diagonal(ρ) * transpose(τ)
    pπ = [τ[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * τ'
    return η
end

# Tabular softmax policy parametrization
function softmaxPolicy(θ)
    θ = reshape(θ, (nA, nO))
    π = exp.(θ)
    for o in 1:nO
        π[:,o] = π[:,o] / sum(π[:,o])
    end
    return π
end

# Reward and state-action frequencies for the softmax model
function softmaxReward(θ, α, β, γ, μ, r)
    π = softmaxPolicy(θ)
    return R(π, α, β, γ, μ, r)
end

function softmaxStateActionFrequency(θ, α, β, γ, μ, r)
    π = softmaxPolicy(θ)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    η = Diagonal(ρ) * transpose(π)
    return η
    #return softmaxStateActionFrequency(π, α, β, γ, μ, r)
end

### Implementation of C-NPG 

# Building blog for safe divergences
ϕ(x) = x * log(x)
dϕ(x) = log(x) + 1
d²ϕ(x) = ((x >= 0) * x)^-1

# Log policy of the model and its Jacobian
logLikelihoods(θ) = log.(softmaxPolicy(θ))
jacobianLogLikelihoods = θ -> ForwardDiff.jacobian(logLikelihoods, θ)

# Preconditionier of Kakade's NPG 
function kakadeConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, β, γ, μ, r)', nS*nA)
    J = jacobianLogLikelihoods(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

# Log state-action frequencies and their Jacobian 
logLikelihoodsSAF(θ) = log.(softmaxStateActionFrequency(θ, α, β, γ, μ, r))
jacobianLogLikelihoodsSAF = θ -> ForwardDiff.jacobian(logLikelihoodsSAF, θ)

# Preconditionier for C-NPG 
function C_NPG_Gramian(θ,λ)
    G = kakadeConditioner(θ)
    J = jacobianLogLikelihoodsSAF(θ)
    H = J' * (vec(c) * vec(c)') * J
    π = softmaxPolicy(θ)
    V = R(π, α, β, γ, μ, c)
    return G + λ * d²ϕ(b - V) * H
end

### Run example 

# Cardinality of the state, action, observation and parameter space
nS = 2;
nA = 2;
nO = 2;
nP = nO*nA;

# State transitions
α = zeros((nS, nS, nA));
α[:,1,:] = Matrix(I, 2, 2);
α[:,2,:] = [0 1; 1 0];

# Observation matrix β is the identity, so we are in a fully observable setting 
β = [1 0; 0 1];  

# Reward and discount 
r = [1. 0.; 2. 0.];
# r = r - ones(size(r)) * sum(r) / length(r)
γ = 0.9;

# Initial distribution 
μ = [0.8, 0.2]

# Define cost and allowed threshold 
c = [10. 0.; 100. 1.] / 100;
b = 0.2;

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, β, γ, μ, r);
∇R = θ -> ForwardDiff.gradient(reward, θ);
    
# Compute the optimal reward by computing it for all 4 deterministic policies 
rewards_det = zeros(2,2);
for i in 1:2
    for j in 1:2
    π_det = transpose([i-1 2-i; j-1 2-j])
    rewards_det[i,j] = R(π_det, α, β, γ, μ, r)
    end
end
R_opt = maximum(rewards_det);
R_min = minimum(rewards_det);

# Create heatmap of the reward in policy space 
n_plot = 100;
x = range(0, 1, length = n_plot);
y = range(0, 1, length = n_plot);
z = zeros(n_plot, n_plot);
for i in 1:n_plot
    for j in 1:n_plot
        π_plot = [x[i] y[j]; 1-x[i] 1-y[j]];
        z[i, j] = R(π_plot, α, β, γ, μ, r) ;
    end
end

p_heatmap = heatmap(x,y,transpose(z));

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, β, γ, μ, r);
∇R = θ -> ForwardDiff.gradient(reward, θ);

# Plot safe policy set 
function C(x,y)
    π = [x y; 1-x 1-y]
    return R(π, α, β, γ, μ, c)
end

x = range(0, 1, length=1000);
y = range(0, 1, length=1000);
cont = @. C(x', y);

### Run multiple trajectories 

# Number of trajectories, iterations and step size 
nTrajectories = 10;
nIterations = 10^4;
Δt = 10^-2;

# Number of different values for 
λs = 10. .^ LinRange(-4,0,3);
λs = append!([0.], λs);
λs = append!(λs, [2.]);

# Optimize using Kakade natural gradient trajectories

# Sample a 10 starting parameters that strictly satisfy the constraint V >= b - 0.1 
θ₀ = zeros(nTrajectories, 4);
for i in 1:nTrajectories
    θ = randn(4)
    π = softmaxPolicy(θ)
    V = R(π, α, β, γ, μ, c)
    while V >= b - 0.1
        θ = randn(4)
        π = softmaxPolicy(θ)
        V = R(π, α, β, γ, μ, c)
    end
    θ₀[i,:] = θ
end

# For plotting 
title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 14, 14, 14;

# Run optimization for different sensitivity values and random initializations 

# Iterate of different sensitivities
for λ in λs
    # Create plot of the heatmap in policy space 
    p = plot(p_heatmap)
    
    # Iterate of different initializations
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        policyTrajectories_Kakade = Float32[];

        # Optimization loop 
        for k in 1:nIterations

            # Check safety 
            π = softmaxPolicy(θ)
            V = R(π, α, β, γ, μ, c)
            if (V > b) && (λ > 0)
                break
            end

            # Store policy 
            append!(policyTrajectories_Kakade, π[1, :])

            # Compute preconditioner 
            if λ == 0.
                G = kakadeConditioner(θ)
            else 
                G = C_NPG_Gramian(θ, λ)
            end

            # Compute update direction, step size and new parameter 
            Δθ = pinv(G) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ

        end
        
        # Rearrange policy trajectories 
        l = div(length(policyTrajectories_Kakade),2)
        policyTrajectories_Kakade = reshape(policyTrajectories_Kakade, (nA,l))
        policyTrajectories_Kakade = policyTrajectories_Kakade'

        # Plot optimization trajectory 
        p = plot(p, policyTrajectories_Kakade[:,1], policyTrajectories_Kakade[:,2], linewidth=5, legend=false, 
            aspect_ratio=:equal, titlefontsize=title_fontsize, tickfontsize=tick_fontsize, 
            legendfontsize=legend_fontsize, guidefontsize=guide_fontsize, fontfamily="Computer Modern", size = (400, 400),
            framestyle=:box, xticks = 0:1:1, yticks = 0:1:1, xlims=(0,1), ylims=(0,1), # title = titles[i], 
            colorrange=(R_min, R_opt), top_margin=10px)

    end

    # Plot safety boundary and export plot 
    p = contour!(x, y, cont, levels=[b], color=:black, linewidth=5, colorbar=false) 
    save("graphics/C-NPG-trajectories-$λ.pdf", p)

end
