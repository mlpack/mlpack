librule(
    name = "multitree_template",
    sources = [],
    headers = ["multitree_dfs.h",
               "multitree_dfs_impl.h",
               "multitree_utility.h"],
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/proximity_project:proximity_project",
               "contrib/dongryel/nested_summation_template:nested_summation_template",
               "mlpack/series_expansion:series_expansion"]
    )
