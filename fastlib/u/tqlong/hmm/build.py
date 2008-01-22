librule(
    name = "hmm",                 # the executable name
    sources = ["support.cc","discreteHMM.cc","gaussianHMM.cc","mixgaussHMM.cc","mixtureDST.cc"],
    headers = ["support.h", "discreteHMM.h","gaussianHMM.h","mixgaussHMM.h","mixtureDST.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "generate",                 # the executable name
    sources = ["generate.cc"],
    deplibs = [":hmm"]       # depends on hmm library in this folder
    )

binrule(
    name = "loglik",                 # the executable name
    sources = ["loglik.cc"],
    deplibs = [":hmm"]       # depends on hmm library in this folder
    )

binrule(
    name = "viterbi",                 # the executable name
    sources = ["viterbi.cc"],
    deplibs = [":hmm"]       # depends on hmm library in this folder
    )

binrule(
    name = "train",                 # the executable name
    sources = ["train.cc"],
    deplibs = [":hmm"]       # depends on hmm library in this folder
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
