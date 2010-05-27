librule(
    name = "2dfactor",                 # the executable name
    sources = ["2dPCA.cc", "2dFactor.cc"],
    headers = ["2dfactor.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "2dfactor_test",                 # the executable name
    sources = ["2dfactor_test.cc"],
    headers = [],
    deplibs = [":2dfactor"]       # depends on hmm library in this folder
    )

librule(
    name = "point_list",                 # the library name
    sources = ["point_list.cc"],
    headers = ["point_list.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "point_list_test",
    sources = ["point_list_test.cc"],
    deplibs = ["fastlib:fastlib_int", ":point_list"] 
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
