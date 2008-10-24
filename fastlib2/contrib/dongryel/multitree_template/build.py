librule(
    name = "multitree_template",
    sources = [],
    headers = ["multitree_common.h",
               "multitree_dfs.h",
               "multitree_dfs_impl.h",
               "upper_triangular_square_matrix.h"],
    deplibs = ["fastlib:fastlib_int",
               "contrib/dongryel/proximity_project:proximity_project",
               "mlpack/series_expansion:series_expansion"]
    )
