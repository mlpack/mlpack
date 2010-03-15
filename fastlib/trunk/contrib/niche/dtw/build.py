binrule(
    name = "dtw",
    sources = ["main.cc"],
    headers = ["dtw.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "multidtw",
    sources = ["multivariate_main.cc"],
    headers = ["dtw.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "test_arraylist",
    sources = ["test_arraylist.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )
