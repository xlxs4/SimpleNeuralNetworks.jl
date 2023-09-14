using SimpleNeuralNetworks
using Documenter

DocMeta.setdocmeta!(SimpleNeuralNetworks, :DocTestSetup, :(using SimpleNeuralNetworks); recursive=true)

makedocs(;
    modules=[SimpleNeuralNetworks],
    authors="Orestis Ousoultzoglou <orousoultzoglou@gmail.com> and contributors",
    repo="https://github.com/xlxs4/SimpleNeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename="SimpleNeuralNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xlxs4.github.io/SimpleNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/xlxs4/SimpleNeuralNetworks.jl",
    devbranch="main",
)
