librule(
    name = "n_point_impl",
    headers = ["n_point_impl.h"],
    sources = ["n_point_impl.cc"],
    deplibs = ["fastlib:fastlib"]
)

librule(
    name = "n_point_alg",
    sources = ["matcher.cc", "n_point.cc", "n_point_nodes.cc", "n_point_perm_free.cc", "perm_free_matcher.cc"],
    headers = ["matcher.h", "n_point.h", "n_point_nodes.h", "n_point_perm_free.h", "perm_free_matcher.h"],
    deplibs = ["fastlib:fastlib", ":n_point_impl"]
)

binrule(
    name = "n_point_testing",
    sources = ["n_point_testing.cc"],
    headers = ["n_point_testing.h"],
    deplibs = ["fastlib:fastlib", ":n_point_alg", ":n_point_impl"]
)

binrule(
    name = "n_point_main",
    sources = ["n_point_main.cc"],
    deplibs = ["fastlib:fastlib", ":n_point_alg"]
)
