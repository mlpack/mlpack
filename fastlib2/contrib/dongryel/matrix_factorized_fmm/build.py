# Library build rule for the matrix-factorized FMM.
librule(
    name = "matrix_factorized_fmm",
    sources = [],
    headers = ["matrix_factorized_fmm.h",
               "matrix_factorized_fmm_impl.h",
               "matrix_factorized_fmm_stat.h",
               "naive_kde.h"],
    deplibs = ["fastlib:fastlib_int",
               "mlpack/series_expansion:series_expansion"]
    )

# Test driver for series expansion library
binrule(
    name = "matrix_factorized_fmm_bin",
    sources = ["matrix_factorized_fmm_main.cc"],
    headers = [],
    deplibs = [":matrix_factorized_fmm"]
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
