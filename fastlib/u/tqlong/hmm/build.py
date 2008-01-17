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
