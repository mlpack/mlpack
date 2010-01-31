# Library build rule for the series expansion implementation
librule(
    name = "series_expansion",
    sources = ["series_expansion_aux.cc"],
    headers = ["farfield_expansion.h",
               "mult_farfield_expansion.h",
               "kernel_aux.h",
               "local_expansion.h",
               "mult_local_expansion.h",
               "mult_series_expansion_aux.h",
               "series_expansion_aux.h",
               "mult_series_expansion_aux.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# Test driver for series expansion library
binrule(
    name = "main",
    sources = ["main.cc"],
    headers = [],
    deplibs = [":series_expansion"]
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
