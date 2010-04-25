librule(
    name = "nested_summation_template",
    sources = [],
    headers = ["function.h",
               "nested_sum_utility.h",
               "operator.h",
               "ratio.h",
               "strata.h",
               "sum.h"],    
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion"]
    )
