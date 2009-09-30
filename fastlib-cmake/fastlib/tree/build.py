# tree is currently an internal-only feature

librule(
    name = "tree",
    sources = [],
    headers = [
        "kdtree.h", "kdtree_impl.h", "bounds.h",
        "spacetree.h", "statistic.h"
    ],
    tests = ["tree_test.cc"],
    deplibs = ["fastlib/base:base", "fastlib/la:la", "fastlib/col:col",
        "fastlib/file:file_int", "fastlib/data:data", "fastlib/fx:fx"]
    )

