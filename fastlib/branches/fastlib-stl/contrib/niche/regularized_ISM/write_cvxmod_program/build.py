binrule(
    name = "write_cvxmod_prog",
    sources = ["write_cvxmod_prog.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib",
               "mlpack/allknn:allknn"]
    )

binrule(
    name = "extract_dists",
    sources = ["extract_dists.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib",
               "mlpack/allknn:allknn"]
    )
