librule(
    name = "thor_kde",                       # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["thor_kde.h"],
    deplibs =
    ["fastlib:fastlib_int",
     "fastlib/thor:thor",
     "contrib/dongryel/multitree_template:multitree_template",
     "contrib/dongryel/nested_summation_template:nested_summation_template",
     "contrib/dongryel/proximity_project:proximity_project",
     "mlpack/series_expansion:series_expansion"]
    )

binrule(
    name = "thor_kde_bin",
    sources = ["thor_kde_main.cc"],
    headers = [],
    deplibs = [":thor_kde"]
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
