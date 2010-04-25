librule(
    name = "optim",                 # the executable name
    sources = ["optim.cc"],
    headers = ["optim.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "optim_test",                 # the executable name
    sources = ["optim_test.cc"],
    headers = ["myfun.h"],
    deplibs = [":optim"]       # depends on hmm library in this folder
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
