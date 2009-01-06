librule(
    name = "nested_summation_template",
    sources = [],
    headers = ["operator.h",
               "ratio.h",
               "sum.h"],
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/multitree_template:multitree_template",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion"]
    )
