librule(
    name = "thor_npc",                       # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["thor_npoint.h", "dfs3.h", "dfs3_impl.h", "two_point.h",
               "three_point.h", "thor_3point.h"],
    deplibs =
    ["fastlib:fastlib_int","fastlib/thor:thor"]     
    )

librule(
    name = "thor_2pc",                       # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["thor_2point.h", "two_point.h"],
    deplibs =
    ["fastlib:fastlib_int","fastlib/thor:thor"]     
    )

binrule(
    name = "thor_2pc_bin",
    sources = ["thor_2pt_main.cc"],
    headers = [],
    deplibs = [":thor_2pc"]
    )

binrule(
    name = "thor_3pc_bin",
    sources = ["thor_3pt_main.cc"],
    headers = [],
    deplibs = [":thor_npc"]
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
