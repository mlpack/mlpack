
binrule(
    name = "allnn_bin",
    sources = ["allnn.cc"],
    linkables = ["fastlib:fastlib_int"])

binrule(
    name = "allnnmpi_bin",
    sources = ["allnn.cc"],
    linkables = ["fastlib:fastlib_int", "par:mpi"],
    cflags = "-DUSE_MPI")
