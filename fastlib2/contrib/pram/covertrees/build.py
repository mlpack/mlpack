librule(
    sources = [],
    headers = ["ctree.h", "cover_tree.h", "allknn.h", "distances.h"],
    tests = ["cover_tree_test.cc"],
    deplibs = ["fastlib:fastlib"]
    )
