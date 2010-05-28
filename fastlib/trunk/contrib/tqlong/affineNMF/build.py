librule(
    name = "anmf",                 # the executable name
    sources = ["imregister.cc", "nmf.cc"],
    headers = ["affineNMF.h", "imregister.h", "nmf.h"],
    deplibs = ["fastlib:fastlib_int"]       # depends on faslib core library
    )

binrule(
    name = "anmf_test",                 # the executable name
    sources = ["ANMF_test.cc"],
    deplibs = [":anmf"]       # depends on affineNMF library in this folder
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
