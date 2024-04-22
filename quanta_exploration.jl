using Agents
using InteractiveDynamics
using GLMakie
using Plots
using Random
using Distributions
using Statistics
using StatsBase
using DataFrames
using DataFramesMeta


function GiniCoefficientJulia(x)
    n = length(x)
    shift = abs(minimum(x, init=0))
    x = x .+ shift
    # If all values are zero, return 0
    if sum(x) == 0
        return 0
    end
    
    # Sort the values
    x = sort(x)
    
    # Calculate the rank of each value
    rank = collect(1:n)
    
    # Calculate the Gini coefficient using the formula
    return (1 / n) * (2 * sum(rank .* x) / sum(x) - (n + 1))
end

seed = 123;

MEAN_MONEY = 60_000
SD_MONEY = 5_000;
SHAPE_PARAMETER_XI = 0.5
MONEY_DISTRIBUTION = GeneralizedPareto(MEAN_MONEY, SD_MONEY, SHAPE_PARAMETER_XI)

data = [ rand(MONEY_DISTRIBUTION,1)[1] for _ in 1:500 ]
Plots.histogram( data, bins=12, title="wealth distribution" )



@agent SchellingAgent GridAgent{2} begin
    remain::Bool 
    group::Int
    money::Float64
    
    potential_energy::Int
    kinetic_energy::Float64 
    
end

seed = 123;

const NN = 13;
const GRID_DIM = (NN,NN);
TOTAL_AGENTS = round( Int, 0.70*prod(GRID_DIM) );

AGENTS_MONEY = rand(MONEY_DISTRIBUTION, TOTAL_AGENTS)
MAX_MONEY = maximum( AGENTS_MONEY );
# MOVEMENT_ENERGY_MIN_REQUIREMENT = 0.2

KineticEnergy(money) = 2 * money / MAX_MONEY
InvKineticEnergy(kinetic_energy) = (1/2) * MAX_MONEY * kinetic_energy #returns the money and money is then effectively the energy
PotentialEnergy(remain) = 1 * Int( !remain )
FIXED_ENERGY_BINS = 0:0.2:1_000;

scheduler_model = Schedulers.Randomly();

# params = Dict(
#     :min_to_be_happy_inner => 4, :min_to_be_happy_side => 3, :min_to_be_happy_corner => 2,
#     :inner_surround => 8, :side_surround => 5, :corner_surround => 3,
    
# )

function initialize(; min_to_be_happy_corner = 2,
    min_to_be_happy_inner = 4, 
    min_to_be_happy_side = 3,
    corner_surround = 3,
    inner_surround = 8, 
    side_surround = 5,
    MOVEMENT_ENERGY_MIN_REQUIREMENT = 2,
    GRID_DIM = GRID_DIM)

    global model_iteration = 0

    space = GridSpaceSingle(GRID_DIM, periodic = false)

    params = Dict(
    :min_to_be_happy_inner => min_to_be_happy_inner, 
    :min_to_be_happy_side => min_to_be_happy_side, 
    :min_to_be_happy_corner => min_to_be_happy_corner,
    :inner_surround => inner_surround, 
    :side_surround => side_surround, 
    :corner_surround => corner_surround, 
    :MOVEMENT_ENERGY_MIN_REQUIREMENT => MOVEMENT_ENERGY_MIN_REQUIREMENT)

    properties = params

    rng = Random.Xoshiro(seed)

    model = UnremovableABM(SchellingAgent, space; rng=rng, properties=properties, 
                            scheduler=scheduler_model)

    for ii in 1:TOTAL_AGENTS
        remain_i = false
        identity_i = ii < TOTAL_AGENTS/2 ? 1 : 2
        money_i = AGENTS_MONEY[ii]
        
        potential_energy_i = PotentialEnergy(remain_i)
        kinetic_energy_i = KineticEnergy(money_i)
        
        agent = SchellingAgent(ii, (1, 1), remain_i, identity_i, money_i,
                                    potential_energy_i, kinetic_energy_i)
        add_agent_single!(agent, model)
    end
    return model
end

model = initialize()

function GetAgentRemainStatus(agent, model, count_near, count_neighbors_same_group)
    
    #agent moves it remain is false and kinetic is higher than potential
    kinetic_higher_than_potential = agent.kinetic_energy > model.MOVEMENT_ENERGY_MIN_REQUIREMENT #PotentialEnergy(agent.remain)
    
    remain = true

    if count_near == model.inner_surround && count_neighbors_same_group >= model.min_to_be_happy_inner
        remain = true
    elseif count_near == model.side_surround && count_neighbors_same_group >= model.min_to_be_happy_side
        remain = true
    elseif count_near == model.corner_surround && count_neighbors_same_group >= model.min_to_be_happy_corner
        remain = true
    else
        remain = false
    end
    
    if( kinetic_higher_than_potential == true && remain == false )
        return false
    else
        return true
    end    
    
end

function MovementMoneySpending(agent, model)
    
    money_spent = InvKineticEnergy(model.MOVEMENT_ENERGY_MIN_REQUIREMENT)
    
    if( money_spent >= agent.money )
        return
    else
        agent.money -=  money_spent
    end
    
    
    agent.kinetic_energy = KineticEnergy(agent.money)
    
    number_of_neighbors = length( collect( nearby_agents( agent, model ) ) )
    neighbor_dividend = money_spent / number_of_neighbors
    # modify to have only 1 agent receive all the quanta of money
    for neighbor in nearby_agents( agent, model )
        neighbor.money += neighbor_dividend
        neighbor.kinetic_energy = KineticEnergy(neighbor.money)
    end
    
end

function MoneySpending(agent, model)
    money_spent = agent.money * 0.05
    
    agent.money -=  money_spent
    agent.kinetic_energy = KineticEnergy(agent.money)
    
end

function agent_step!(agent, model)
    
    count_near = 0
    for pos in nearby_positions( agent.pos , model )
        count_near += 1
    end
    
    count_neighbors_same_group = 0    
    for neighbor in nearby_agents( agent, model )
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    
    agent.remain = GetAgentRemainStatus(agent, model, count_near, count_neighbors_same_group) 
    agent.potential_energy = PotentialEnergy( agent.remain )
    
    if agent.remain == false
        #move to random position
        move_agent_single!(agent, model)
        
        #money held by agent changes
        MovementMoneySpending(agent, model)
        
    end
    #money changes by agent time
    MoneySpending(agent, model)
end

function ModelRemainTrajectory(model)
    remain = 0
    for agent in allagents(model)
        remain += agent.remain 
    end
    return remain / nagents(model)
end

function ModelFinancialDisparity(model)
    disparity = 0
    for agent in allagents(model)
        
        disparity_temp = 0
        number_of_neighbors = length( collect( nearby_agents(agent, model, 1) ) )
        
        for neighbor in nearby_agents(agent, model, 1)
            disparity_temp += abs( agent.money - neighbor.money )
        end
        
        if( number_of_neighbors > 0 )
            disparity_temp = disparity_temp / number_of_neighbors
            disparity += disparity_temp
        end

    end
    return disparity
end

function ModelEntropy(model)
    
    agent_energies = Float64[]
    
    for agent in allagents(model)
        agent_energy = agent.kinetic_energy + agent.potential_energy 
        push!(agent_energies, agent_energy)
    end
    energy_hist = fit(Histogram, agent_energies, FIXED_ENERGY_BINS)
    energy_weights = energy_hist.weights
    nonzero_weights = filter(!iszero, energy_weights)
    energy_probabilities = nonzero_weights ./ sum(nonzero_weights)
    
    SS = 0
    for ii in 1:length( energy_probabilities )
        SS += (-1) * energy_probabilities[ii] * log( energy_probabilities[ii] )
    end
    
    return SS
end

function ModelInternalEnergy(model)
    # U = sum_distinct_states_i p_i * E_i
    
    agent_energies = Float64[]
    
    for agent in allagents(model)
        agent_energy = agent.kinetic_energy + agent.potential_energy
        push!(agent_energies, agent_energy)
    end
    
    energy_hist = fit(Histogram, agent_energies, FIXED_ENERGY_BINS)
    energy_probabilities = energy_hist.weights ./ sum(energy_hist.weights)
    energy_edges = collect( energy_hist.edges[1] )
    energy_values = [ (energy_edges[i]+energy_edges[i+1])/2 for i in 1:length(energy_edges)-1 ]
    
    UU = 0
    for ii in 1:length(energy_values)
        UU += energy_probabilities[ii] * energy_values[ii]
    end
    
    return UU
end

agent_money_gain = 100_000

function model_step!(model)
    global model_iteration += 1
    
    if( model_iteration % 100 == 0 && model_iteration != 0 )
        for agent in allagents(model)
            agent.money += agent_money_gain
            agent.kinetic_energy = KineticEnergy(agent.money)
        end        
    end
    
end

adata = [ :potential_energy, :kinetic_energy, :money ]
mdata = [ ModelRemainTrajectory, ModelFinancialDisparity, 
                        ModelEntropy, ModelInternalEnergy ]



parameters = Dict(
    :MOVEMENT_ENERGY_MIN_REQUIREMENT => [0.1, 0.5],
    :min_to_be_happy_corner => (2),
    :min_to_be_happy_inner => (4), 
    :min_to_be_happy_side => (3),
    :corner_surround => (3),
    :inner_surround => (8), 
    :side_surround => (5), 
    
)


InvKineticEnergy(2)

model = initialize()

agent_df, model_df = paramscan(parameters, initialize; adata = adata, mdata = mdata, agent_step! = agent_step!, 
                                model_step! = model_step!, n=300)


p1 = Plots.plot( model_df[!,:ModelRemainTrajectory], linewidth=6, legend=false, 
title="Remain Ratio", xlabel="iteration" )
p2 = Plots.plot( model_df[!,:ModelFinancialDisparity], linewidth=6, legend=false,
            title="Monetary Disparity \n (aggregate)", xlabel="iteration" )
p3 = Plots.plot( model_df[!,:ModelEntropy], linewidth=6, legend=false,
            title="Model Entropy", xlabel="iteration" )
p4 = Plots.plot( model_df[!,:ModelInternalEnergy], linewidth=6, legend=false,
            title="Model Internal Energy", xlabel="iteration" )

pOveral = Plots.plot([p1,p2,p3,p4]..., layout=grid(2,2) )

step_num = 300;

gini_internal_U = []
gini_kinetic = []
percentage_remain = []
kinetic_means = []




agent_df_first = @subset(agent_df, :MOVEMENT_ENERGY_MIN_REQUIREMENT .== 0.1)
agent_df_second = @subset(agent_df, :MOVEMENT_ENERGY_MIN_REQUIREMENT .== 0.5)

for step in 1:step_num
    agent_U_vals = agent_df_first[isequal.(agent_df_first.step, step), :][!,:potential_energy]
    gini = GiniCoefficientJulia( agent_U_vals )
    push!(gini_internal_U,gini)
    push!(percentage_remain, length( findall( agent_U_vals .== 0 ) ) / TOTAL_AGENTS )
    
    agent_K_vals = agent_df_first[isequal.(agent_df_first.step, step), :][!,:kinetic_energy]
    agent_K_groups = (KineticEnergy(MAX_MONEY)/100) .* div.(agent_K_vals, KineticEnergy(MAX_MONEY)/100 ) 
    
    if( length(findall( agent_K_vals .< 0 )) > 0 )
        println("BAD")
        println(agent_K_vals)
    end
    gini = GiniCoefficientJulia( agent_K_groups )
    push!(gini_kinetic,gini)
    push!(kinetic_means, mean(agent_K_vals))
end

length(gini_internal_U)

p1_gini = Plots.plot(gini_internal_U, linewidth=6, legend=false,
            title="Gini Coefficient\nPotential Energy", xlabel="iteration",
            ylim=ylim=[0,maximum(gini_internal_U)*1.1])
p2_gini = Plots.plot(gini_kinetic, linewidth=6, legend=false,
                title="Gini Coefficient\nKinetic Energy", xlabel="iteration",
                ylim=[0,maximum(gini_kinetic)*1.1])
p3_gini = Plots.plot(percentage_remain, linewidth=6, legend=false,
                title="Agent Remain Percentage", xlabel="iteration" )
p4_gini = Plots.plot(kinetic_means, linewidth=6, legend=false,
                title="Kinetic Energy Means", xlabel="iteration" )
p_gini = Plots.plot([p1_gini,p2_gini,p3_gini,p4_gini]...,layout=grid(2,2))
# savefig(p_gini,"./plots4/giniEnergies.pdf")
# display(p_gini)









gini_internal_U2 = []
gini_kinetic2 = []
percentage_remain2 = []
kinetic_means2 = []

for step in 1:step_num
    agent_U_vals2 = agent_df_second[isequal.(agent_df_second.step, step), :][!,:potential_energy]
    gini2 = GiniCoefficientJulia( agent_U_vals2 )
    push!(gini_internal_U2,gini2)
    push!(percentage_remain2, length( findall( agent_U_vals2 .== 0 ) ) / TOTAL_AGENTS )
    
    agent_K_vals2 = agent_df_second[isequal.(agent_df_second.step, step), :][!,:kinetic_energy]
    agent_K_groups2 = (KineticEnergy(MAX_MONEY)/100) .* div.(agent_K_vals2, KineticEnergy(MAX_MONEY)/100 ) 
    
    if( length(findall( agent_K_vals2 .< 0 )) > 0 )
        println("BAD")
        println(agent_K_vals2)
    end
    gini = GiniCoefficientJulia( agent_K_groups2 )
    push!(gini_kinetic2,gini)
    push!(kinetic_means2, mean(agent_K_vals2))
end

p1_gini = Plots.plot(gini_internal_U2, linewidth=6, legend=false,
            title="Gini Coefficient\nPotential Energy", xlabel="iteration",
            ylim=ylim=[0,maximum(gini_internal_U)*1.1])
p2_gini = Plots.plot(gini_kinetic2, linewidth=6, legend=false,
                title="Gini Coefficient\nKinetic Energy", xlabel="iteration",
                ylim=[0,maximum(gini_kinetic)*1.1])
p3_gini = Plots.plot(percentage_remain2, linewidth=6, legend=false,
                title="Agent Remain Percentage", xlabel="iteration",
                ylim=ylim=[0,1.1] )
p4_gini = Plots.plot(kinetic_means2, linewidth=6, legend=false,
                title="Kinetic Energy Means", xlabel="iteration" )
p_gini = Plots.plot([p1_gini,p2_gini,p3_gini,p4_gini]...,layout=grid(2,2))


using CSV

#save dataframe to CSV file
#MDATA is the model data (parameters of the mode / complex aggregates)

CSV.write("C:\\Users\\tyler\\OneDrive\\Desktop\\output.csv", agent_df_first)