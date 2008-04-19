
librule(
    name = "proximity_project",              # this line can be safely omitted
    sources = ["gen_range.cc"],              # files that must be compiled
    headers = ["subspace_stat.h",
               "general_type_bounds.h",
               "gen_range.h",
               "gen_kdtree.h",
               "gen_kdtree_impl.h",
               "gen_metric_tree.h",
               "gen_metric_tree_impl.h",
               "general_spacetree.h"],       # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "proximity_project_bin",          # the executable name
    sources = ["main.cc"],                   # compile multibody.cc
    headers = [],                            # no extra headers
    deplibs = [":proximity_project",
               "fastlib:fastlib_int"]
    )

binrule(
    name = "kfold_splitter_bin",
    sources = ["kfold_splitter.cc"],
    headers = ["kfold_splitter.h"],
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
