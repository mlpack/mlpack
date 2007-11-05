# tree is currently an internal-only feature

librule(
    sources = [],
    headers = [
        "kdtree.h", "kdtree_impl.h", "bounds.h",
        "spacetree.h", "statistic.h"
    ],
    tests = ["tree_test.cc"],
    deplibs = ["base:base", "la:la", "col:col",
        "file:file_int", "data:data", "fx:fx"]
    )

