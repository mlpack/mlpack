librule(name="allknn",
    headers=["allknn.h"],
    tests=["allknn_test.cc"],
    deplibs=["fastlib:fastlib"]
    );

binrule(name="allknn_exe",
    sources=["main.cc"],
    deplibs=["fastlib:fastlib", ":allknn"])
