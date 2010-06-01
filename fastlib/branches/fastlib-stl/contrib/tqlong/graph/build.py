librule(
    name = "graph",                 # the executable name
    #sources = ["support.cc","discreteHMM.cc","gaussianHMM.cc","mixgaussHMM.cc","mixtureDST.cc"],
    #headers = ["support.h", "discreteHMM.h","gaussianHMM.h","mixgaussHMM.h","mixtureDST.h"],
    sources = ["graph.cc"],
    headers = ["graph.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "maxflow",                 # the executable name
    sources = ["maxflow.cc"],
    deplibs = [":graph"]       # depends on hmm library in this folder
    )
