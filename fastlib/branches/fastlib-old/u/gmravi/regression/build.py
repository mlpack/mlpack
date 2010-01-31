binrule(
    name = "main_final",                     # the executable name
    sources = ["main_final.cc"],                   
    headers = ["regression_vector.h","regression_matrix.h"], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
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
