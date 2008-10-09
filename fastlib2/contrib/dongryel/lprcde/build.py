librule(
    name = "lprcde",                         # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["nwrcde_common.h",
               "nwrcde_delta.h",
               "nwrcde.h",
               "nwrcde_impl.h",
               "nwrcde_query_postponed.h",
               "nwrcde_query_result.h",
               "nwrcde_query_summary.h",
               "nwrcde_stat.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion"]
    )

binrule(
    name = "nwrcde_bin",
    sources = ["nwrcde_main.cc"],
    headers = [],
    deplibs = [":lprcde"]
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
