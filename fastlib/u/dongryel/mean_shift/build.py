
librule(
    name = "mean_shift",                     # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["mean_shift.h"],              # include files part of the 'lib'
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "mean_shift_bin",                 # the executable name
    sources = ["main.cc"],                   # compile multibody.cc
    headers = [],                            # no extra headers
    deplibs = [":mean_shift",
               "u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
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
