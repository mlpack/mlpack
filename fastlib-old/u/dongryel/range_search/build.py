
librule(
    name = "ortho_range_search",
    sources = [],
    headers = ["ortho_range_search.h",
               "range_reader.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

binrule(
    name = "ortho_range_search_bin",
    sources = ["ortho_range_search_main.cc"],
    headers = [],
    deplibs = [":ortho_range_search",
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
