using SimpleNeuralNetworks
using Test
using Aqua

@testset "SimpleNeuralNetworks.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SimpleNeuralNetworks; ambiguities = false)
    end
    @testset "relu" begin
        @test relu(0.0) == 0.0
        @test relu(1.0) == 1.0
        @test relu(-1.0) == 0.0
        @test isnan(relu(NaN32))
        @test !isnan(relu(Inf32))
        @test !isnan(relu(-Inf32))
    end
    @testset "Softmax" begin
        @test softmax(Int[0, 0]) == [0.5, 0.5]
        xs = rand(5, 5)
        @test all(sum(softmax(xs), dims = 1) .≈ 1)
        @test all(sum(softmax(xs; dims = 2), dims = 2) .≈ 1)
        @test sum(softmax(vec(xs))) ≈ 1
    end
    @testset "_nfan" begin
        @test SimpleNeuralNetworks._nfan() == (1, 1)
        @test SimpleNeuralNetworks._nfan(4) == (1, 4)
        @test SimpleNeuralNetworks._nfan(4, 5) == (5, 4)
    end
    @testset "zeros32" begin
        @test size(zeros32(3)) == (3,)
        @test size(zeros32(3, 4)) == (3, 4)
        @test size(zeros32(3, 4, 5)) == (3, 4, 5)
        @test eltype(zeros32(3)) == Float32
    end
    @testset "he_normal" begin
        @test size(he_normal(3)) == (3,)
        @test size(he_normal(3, 4)) == (3, 4)
        @test eltype(he_normal(3)) == Float32
    end
    @testset "xlogy" begin
        @test iszero(SimpleNeuralNetworks.xlogy(0, 1))
        @test isnan(SimpleNeuralNetworks.xlogy(NaN, 1))
        @test isnan(SimpleNeuralNetworks.xlogy(1, NaN))
        @test isnan(SimpleNeuralNetworks.xlogy(NaN, NaN))
        @test SimpleNeuralNetworks.xlogy(2, 3) ≈ 2.0 * log(3.0)
        @inferred SimpleNeuralNetworks.xlogy(2, 3)
        @inferred SimpleNeuralNetworks.xlogy(0, 1)
    end
    @testset "crossentropy" begin
        @test crossentropy([0.1, 0.0, 0.9], [0.1, 0.0, 0.9]) ≈
              crossentropy([0.1, 0.9], [0.1, 0.9])
    end
    @testset "Chain" begin
        @test_nowarn Chain(Dense(10 => 5, relu), Dense(5 => 2), softmax)(randn(Float32, 10))
    end
    @testset "Dense" begin
        @test Dense(3 => 4).activation == identity
        @test size(Dense(3 => 4).weight) == (4, 3)
        @test size(Dense(3 => 4).bias) == (4,)
        @test Dense(3 => 4, relu).activation == relu
    end
end
