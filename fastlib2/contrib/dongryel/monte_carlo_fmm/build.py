# Library build rule for the Monte Carlo FMM.
librule(
    name = "monte_carlo_fmm",
    sources = [],
    headers = ["monte_carlo_fmm.h",
               "monte_carlo_fmm_impl.h",
               "inverse_normal_cdf.h",
               "naive_kde.h"],
    deplibs = ["fastlib:fastlib_int",
               "mlpack/series_expansion:series_expansion",
               "contrib/dongryel/proximity_project:proximity_project"]
    )

# Test driver for the Monte Carlo FMM
binrule(
    name = "monte_carlo_fmm_bin",
    sources = ["monte_carlo_fmm_main.cc"],
    headers = [],
    deplibs = [":monte_carlo_fmm"]
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
