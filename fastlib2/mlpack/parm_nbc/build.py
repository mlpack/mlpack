
librule(
    name = "simple_nbc",              # this line can be safely omitted
    #sources = [],       # files that must be compiled
    headers = ["simple_nbc.h","phi.h","math_functions.h"], # include files part of the 'lib'
    deplibs = ["fastlib:fastlib"],  # depends on fastlib core
    tests = ["test_simple_nbc_main.cc"]
    )

binrule(
    name = "simple_nbc_main",                 # the executable name
    sources = ["nbc_main.cc"],         # compile main.cc
    #headers = ["nbc_main.h"],                  # no extra headers
    deplibs = [":simple_nbc","fastlib:fastlib"]       # depends on example in this folder
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
