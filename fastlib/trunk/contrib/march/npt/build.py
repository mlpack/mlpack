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

librule(
    name = "n_point_multi",
    sources = ["n_point_multi.cc", "results_tensor.cc", "multi_matcher.cc", "n_point_nodes.cc"],
    headers = ["n_point_multi.h", "results_tensor.h", "multi_matcher.h", "n_point_nodes.h"],
    deplibs = ["fastlib:fastlib", ":n_point_impl"]
)

binrule(
    name = "n_point_naive_multi_main",
    sources = ["n_point_naive_multi_main.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib", ":n_point_alg"],
)

binrule( 
    name = "make_matchers",
    sources = ["make_matchers.cc", "results_tensor.cc", "n_point_impl.cc"],
    headers = ["results_tensor.h", "n_point_impl.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "n_point_multi_main",
    sources = ["n_point_multi_main.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib", ":n_point_multi"]
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
