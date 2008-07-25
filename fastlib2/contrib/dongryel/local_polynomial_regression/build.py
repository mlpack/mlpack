# The library build rule for batch SVD factorization based local
# polynomial regression.

librule(
    name = "svd_lpr",
    sources = [],
    headers = ["svd_lpr.h",
               "svd_lpr_user_level_impl.h"],
    deplibs = ["contrib/dongryel/proximity_project:proximity_project",
               "fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "svd_lpr_bin",
    sources = ["svd_lpr_main.cc"],
    headers = [],
    deplibs = [":svd_lpr",
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
