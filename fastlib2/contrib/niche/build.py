binrule(
    name = "test2",
    sources = ["test2.cc"],
    headers = [],
    deplibs = ["contrib/dongryel/regression:krylov_lpr",
               "fastlib:fastlib_int",
               "fastlib/sparse/trilinos:trilinos"]
    )
