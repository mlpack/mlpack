librule(
    name = "dense_lpr",
    sources = [],
    headers = ["dense_lpr.h",
               "dense_lpr_impl.h",
               "epan_kernel_moment_info.h",
               "lpr_util.h",
               "matrix_util.h",
               "multi_index_util.h",
               "quick_prune_lpr.h",
               "relative_prune_lpr.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The library build rule for Krylov-subspace based local polynomial
# regression.

librule(
    name = "krylov_lpr",
    sources = [],
    headers = ["epan_kernel_moment_info.h",
               "krylov_lpr.h",
               "krylov_lpr_setup_impl.h",
               "krylov_lpr_test.h",
               "lpr_util.h",
               "naive_lpr.h"],
    deplibs = ["fastlib:fastlib_int",
               "fastlib/sparse/trilinos:libtrilinos"]        # dependency
    )

# The binary executable rule for Krylov-subspace based local
# polynomial regression.
binrule(
    name = "krylov_lpr_bin",
    sources = ["krylov_lpr_main.cc"],
    headers = [],
    deplibs = [":krylov_lpr",
               "fastlib:fastlib_int",
               "fastlib/sparse/trilinos:libtrilinos"]
    )

binrule(
    name = "dense_lpr_bin",
    sources = ["dense_lpr_main.cc"],
    headers = [],
    deplibs = [":dense_lpr",
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
