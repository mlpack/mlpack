librule(
    name = "fast_multipole_method",          # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["fast_multipole_method.h",
               "fmm_stat.h"],                # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/proximity_project:proximity_project"]    
    )

binrule(
    name = "fast_multipole_method_bin",
    sources = ["fast_multipole_method_main.cc"],
    headers = [],
    deplibs = [":fast_multipole_method"]
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
