librule(
    name = "continuous_fmm",
    sources = [],
    headers = ["continuous_fmm.h",
               "fmm_stat.h"],
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/multitree_template:multitree_template",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion", 
               "contrib/march/fock_matrix/fock_impl:eri"]
    )

librule(
    name = "fast_multipole_method",          # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["fast_multipole_method.h",
               "fmm_stat.h"],                # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/multitree_template:multitree_template",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion"]
    )

librule(
    name = "chebyshev_fit",
    sources = [],
    headers = ["chebyshev_fit.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

librule(
    name = "multibody_force",
    sources = [],
    headers = ["multibody_force_problem.h",
               "multibody_kernel.h"],    
    deplibs = ["contrib/dongryel/multitree_template:multitree_template",
               "contrib/dongryel/nested_summation_template:nested_summation_template"]
    )

librule(
    name = "multibody_potential",
    sources = [],
    headers = ["at_potential_kernel.h",              
               "mbp_delta.h",
               "mbp_global.h",
               "mbp_kernel.h",
               "mbp_query_postponed.h",
               "mbp_query_result.h",
               "mbp_query_summary.h",
               "mbp_stat.h",
               "multibody_potential_problem.h",
               "three_body_gaussian_kernel.h"],
    deplibs = ["contrib/dongryel/multitree_template:multitree_template",
               "contrib/dongryel/nested_summation_template:nested_summation_template",
               ":chebyshev_fit"]
    )

binrule(
    name = "fast_multipole_method_bin",
    sources = ["fast_multipole_method_main.cc"],
    headers = [],
    deplibs = [":fast_multipole_method"]
    )

binrule(
    name = "continuous_fmm_bin",
    sources = ["cfmm_main.cc"],
    headers = [],
    deplibs = [":continuous_fmm"]
    )

binrule(
    name = "multibody_force_bin",
    sources = ["multibody_force_main.cc"],
    headers = [],
    deplibs = [":multibody_force"]
    )

binrule(
    name = "multibody_potential_bin",
    sources = ["multibody_potential_main.cc"],
    headers = [],
    deplibs = [":multibody_potential"]
    )

binrule(
    name = "chebyshev_fit_test",
    sources = ["chebyshev_fit_test.cc"],
    headers = [],
    deplibs = [":chebyshev_fit"]
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
