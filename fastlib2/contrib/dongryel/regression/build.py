# The library build rule for Krylov-subspace based local linear
# regression.
librule(
    name = "local_linear_krylov",
    sources = [],
    headers = ["local_linear_krylov.h",
               "local_linear_krylov_setup_impl.h",
               "local_linear_krylov_solver_impl.h",
               "local_linear_krylov_test.h",
               "naive_lpr.h"],
    deplibs = ["mlpack/series_expansion:series_expansion",
               "fastlib:fastlib_int"]        # dependency
    )

# The binary executable rule for Krylov-subspace based local linear
# regression.
binrule(
    name = "local_linear_krylov_bin",
    sources = ["local_linear_krylov_main.cc"],
    headers = [],
    deplibs = [":local_linear_krylov",
               "mlpack/series_expansion:series_expansion",
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
