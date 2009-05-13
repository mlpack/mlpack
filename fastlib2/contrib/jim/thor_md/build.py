librule(
    name = "thor_md",                       # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["thor_md.h", "dfs3.h", "dfs3_impl.h", "two_body.h",
               "three_body.h"],
    deplibs =
    ["fastlib:fastlib_int","fastlib/thor:thor"]     
    )

binrule(
    name = "thor_md_bin",
    sources = ["thor_md_main.cc"],
    headers = [],
    deplibs = [":thor_md"]
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
