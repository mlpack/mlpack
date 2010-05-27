librule(name="allknn_balltree",
    headers=["allknn_balltree.h"],
    tests=["allknn_balltree_test.cc"],
    deplibs=["fastlib:fastlib"]
    );

binrule(name="bt_knn",
    sources=["allknn_balltree_test.cc"],
    deplibs=["fastlib:fastlib", ":allknn_balltree"]);

binrule(name="all_knn_graph",
    sources=["allknn_balltree_construct_graph.cc"],
    deplibs=["fastlib:fastlib", ":allknn_balltree"]);
