librule(
    name = "blah",
    sources = [],
    headers = ["ctree_gc.h", "ctree_impl.h", "cover_tree_gc.h", "allknn_gc.h", "allknn_impl.h", "distances_gc.h", "gonzalez_gc.h"],
    #tests = ["cover_tree_test_gc.cc"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "main",
    sources = ["cover_tree_test_gc.cc"],
    headers = [],
    deplibs = [":blah"]
    )
