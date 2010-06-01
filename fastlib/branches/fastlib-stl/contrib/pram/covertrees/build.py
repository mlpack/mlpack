librule(
    sources = [],
    headers = ["ctree.h", "ctree_impl.h", "cover_tree.h", "allknn.h", "allknn_impl.h", "distances.h"],
    tests = ["cover_tree_test.cc"],
    deplibs = ["fastlib:fastlib"]
    )

