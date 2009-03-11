
librule(
    name = "lle",                    # this line can be safely omitted
    sources = [],                    # files that must be compiled
    headers = ["lle.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib"],   # depends on fastlib core
    #tests = ["lle_tests.cc"]
    )

binrule(
    name = "lle_main",                   # the executable name
    sources = ["lle_main.cc"],           # compile main.cc
    #headers = [],                       # no extra headers
    deplibs = [":lle", "fastlib:fastlib"]#
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
