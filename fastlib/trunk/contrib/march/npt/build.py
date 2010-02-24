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

librule(
    name = "n_point_alg",
    sources = ["matcher.cc", "n_point.cc"],
    headers = ["matcher.h", "n_point.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "n_point_main",
    sources = ["n_point_main.cc"],
    deplibs = ["fastlib:fastlib", ":n_point_alg"]
)
