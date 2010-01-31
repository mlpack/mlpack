binrule(
    name = "parallel_em",           # the executable name
    sources = ["fl_par_em.cc", "fl_data_io.cc", "fl_kmeans.cc"],
    headers = ["fl_par_em.h", "fl_data_io.h", "fl_kmeans.h"],
    linkables = ["fastlib:fastlib"]
    )


binrule(
    name = "csv2bin",           # the executable name
    sources = ["csv2bin.cc", "fl_data_io.cc"],
    headers = ["fl_data_io.h"],
    linkables = ["fastlib:fastlib"]
    )
