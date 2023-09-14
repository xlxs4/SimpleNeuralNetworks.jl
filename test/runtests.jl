using SimpleNeuralNetworks
using Test
using Aqua

@testset "SimpleNeuralNetworks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SimpleNeuralNetworks)
    end
    # Write your tests here.
end
