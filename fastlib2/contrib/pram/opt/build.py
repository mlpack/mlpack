
#librule(
    #name = "mog_em",                    # this line can be safely omitted
    #sources = [],            # files that must be compiled
    #headers = [],# include files part of the 'lib'
    #deplibs = ["fastlib:fastlib"], # depends on fastlib core
    #tests = ["mog_em_tests.cc"]        # this file contains a main with test functions
    #)

binrule(
    name = "main",                  # the executable name
    sources = ["main.cc"],          # compile main.cc
    headers = ["phi.h","optimizers_reloaded.h"],           # no extra headers
    deplibs = ["fastlib:fastlib"]           #
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
