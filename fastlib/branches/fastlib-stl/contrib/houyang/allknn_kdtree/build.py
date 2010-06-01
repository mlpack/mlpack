librule(name="allknn_kdtree",
    headers=["allknn_kdtree.h"],
    tests=["allknn_kdtree_test.cc"],
    deplibs=["fastlib:fastlib"]
    );

binrule(name="kdt_knn",
    sources=["allknn_kdtree_test.cc"],
    deplibs=["fastlib:fastlib", ":allknn_kdtree"]);

binrule(name="all_knn_graph",
    sources=["allknn_kdtree_construct_graph.cc"],
    deplibs=["fastlib:fastlib", ":allknn_kdtree"]);
