librule(
    name = "naive_two_point",
    sources = [],
    headers = ["naive_two_point.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "n_point_testing",
    sources = ["n_point_testing.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib", ":naive_two_point"]
)
