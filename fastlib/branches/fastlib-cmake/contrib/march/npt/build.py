librule(
    name = "n_point_alg",
    sources = ["matcher.cc", "n_point.cc"],
    headers = ["matcher.h", "n_point.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "n_point_testing",
    sources = ["n_point_testing.cc"],
    headers = ["n_point_testing.h"],
    deplibs = ["fastlib:fastlib", ":n_point_alg"]
)

binrule(
    name = "n_point_main",
    sources = ["n_point_main.cc"],
    deplibs = ["fastlib:fastlib", ":n_point_alg"]
)
