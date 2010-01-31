
#The library rule for Local Polynomial expansion
librule(
    name = "naive_kde_local_polynomial",            # this line can be safely omitted
    sources = [""],                         # files that must be compiled
    headers = ["naive_kde_local_polynomial.h","vector_kernel.h","naive_local_likelihood.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]       # depends on fastlib core
    )


#The binary rule for KNN Based regression
binrule(
    name = "naive_kde_lp",                     # the executable name
    sources = ["naive_kde_local_polynomial_main.cc"],                   
    headers = ["naive_kde_local_polynomial.h","vector_kernel.h","naive_local_likelihood.h","naive_kde.h",], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
    )

#The library rule for Cross Validation
librule(
    name = "density_crossvalidation",            # this line can be safely omitted
    sources = [""],                         # files that must be compiled
    headers = ["cross_validation.h","naive_kde.h","naive_kde_local_polynomial.h","naive_local_likelihood.h"],      # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]       # depends on fastlib core
    )


#The binary rule for KNN Based regression
binrule(
    name = "crossvalidation",                     # the executable name
    sources = ["crossvalidation_density_estimates_main.cc"],                   
    headers = ["cross_validation.h","naive_kde_local_polynomial.h","naive_kde.h","naive_local_likelihood.h"], # no extra headers
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
