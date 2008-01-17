binrule(
    name = "main",                 # the executable name
    sources = ["main.cc","discreteHMM.cc","support.cc","discreteDST.cc"],         
    headers = ["discreteHMM.h", "support.h","discreteDST.h"],              
    deplibs = ["fastlib:fastlib_int"]       # depends on example in this folder
    )

binrule(
    name = "gen",                 # the executable name
    sources = ["main.cc","discreteDST.cc","support.cc"],         
    headers = ["discreteHMM.h", "support.h", "gaussianHMM.h"],              
    deplibs = ["fastlib:fastlib_int"]       # depends on example in this folder
    )

binrule(
    name = "mix",                 # the executable name
    sources = ["main.cc","support.cc","mixgaussHMM.cc","mixtureDST.cc", "gaussianHMM.cc"],         
    headers = ["support.h", "mixgaussHMM.h","mixtureDST.h", "gaussianHMM.h"],              
    deplibs = ["fastlib:fastlib_int"]       # depends on example in this folder
    )

binrule(
    name = "mixgen",                 # the executable name
    sources = ["main.cc","support.cc","mixgaussHMM.cc","mixtureDST.cc",  "gaussianHMM.cc"],         
    headers = ["support.h", "mixgaussHMM.h","mixtureDST.h", "gaussianHMM.h"],              
    deplibs = ["fastlib:fastlib_int"]       # depends on example in this folder
    )

binrule(
    name = "hmm",                 # the executable name
    sources = ["hmm.cc","support.cc","discreteHMM.cc","gaussianHMM.cc","mixgaussHMM.cc","mixtureDST.cc"],
    headers = ["support.h", "discreteHMM.h","gaussianHMM.h","mixgaussHMM.h","mixtureDST.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on example in this folder
    )


# to build:
# 1. make sure have environment variables set up:
#    $ source /full/path/to/fastlib/script/fl-env /full/path/to/fastlib
#    (you might want to put this in bashrc)
# 2. fl-build main
#    - this automatically will assume --mode=check, the default
#    - type fl-build --help for help
# 3. ./main
#    - to build same target again, type: make
#    - to force recompilation, type: make clean
