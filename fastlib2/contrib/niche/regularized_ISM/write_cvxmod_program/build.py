binrule(
    name = "writeprog",
    sources = ["main.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib",
               "mlpack/allknn:allknn"]
    )
