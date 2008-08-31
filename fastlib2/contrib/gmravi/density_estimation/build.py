
#The library rule for Local Likelihood density estimation
librule(
    name = "fast_llde_lib",            # this line can be safely omitted
    sources = [""],                         # files that must be compiled
    headers = ["fast_llde.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]       # depends on fastlib core
    )


#The binary rule for Local Likelihood density estimation
binrule(
    name = "fast_llde",                     # the executable name
    sources = ["fast_llde_main.cc"],                   
    headers = ["fast_llde.h"], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
    )


 #The library rule for multi-dimensional local likelihood calculations

librule(
    name = "fast_llde_multi_lib",            # this line can be safely omitted
    sources = [""],                         # files that must be compiled
    headers = ["fast_llde_multi.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]       # depends on fastlib core
    )


#The binary rule for Local Likelihood density estimation
binrule(
    name = "fast_llde_multi",                     # the executable name
    sources = ["fast_llde_multi_main.cc"],                   
    headers = ["fast_llde_multi.h"], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
    )


#The library rule for average over datasets

librule(
    name = "avod_lib",            # this line can be safely omitted
    sources = [""],                         # files that must be compiled
    headers = ["fast_llde_multi.h","average_over_datasets.h","naive_kde.h"],             # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]       # depends on fastlib core
    )


#The binary rule for Local Likelihood density estimation
binrule(
    name = "avod",                     # the executable name
    sources = ["average_over_datasets_main.cc"],                   
    headers = ["fast_llde_multi.h","average_over_datasets.h","naive_kde.h"], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
    )
