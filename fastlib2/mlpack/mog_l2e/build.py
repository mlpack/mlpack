
librule(
    name = "mog_l2e",                    # this line can be safely omitted
    sources = ["mog.cc"],                # files that must be compiled
    headers = ["mog.h","phi.h"],         # include files part of the 'lib'
    deplibs = ["fastlib:fastlib"]        # depends on fastlib core
    #tests = ["mog_l2e_tests.cc"]
    )

binrule(
    name = "mog_l2e_main",                  # the executable name
    sources = ["mog_l2e_main.cc"],          # compile main.cc
    #headers = [,                           # no extra headers
    deplibs = [":mog_l2e","fastlib:fastlib"]#
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
