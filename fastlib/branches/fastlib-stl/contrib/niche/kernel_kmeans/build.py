binrule(
    name = "kernel_kmeans",
    sources = ["main.cc"],
    headers = ["kernel_kmeans.h","../../../mlpack/fastica/lin_alg.h"],
    deplibs = ["fastlib:fastlib"]
)
