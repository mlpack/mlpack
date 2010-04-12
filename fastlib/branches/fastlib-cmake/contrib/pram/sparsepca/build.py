
librule(
    name = "sparse_pca",                    # this line can be safely omitted
    sources = ["sparsepca.cc"],            # files that must be compiled
    headers = ["sparsepca.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib"],         # depends on fastlib core
    #tests = ["sparsepca_tests.cc"]        # this file contains a main with test functions
    )

binrule(
    name = "spca_main",                  # the executable name
    sources = ["spca_main.cc"],          # compile main.cc
    #headers = [],                       # no extra headers
    deplibs = [":sparse_pca", "fastlib:fastlib"]    #
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
