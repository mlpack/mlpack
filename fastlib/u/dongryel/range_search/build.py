
librule(
    name = "range_search",                   # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["ortho_range_search.h",
               "range_reader.h"],            # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "range_search_bin",               # the executable name
    sources = ["main.cc"],                   # compile multibody.cc
    headers = [],                            # no extra headers
    deplibs = [":range_search",
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
