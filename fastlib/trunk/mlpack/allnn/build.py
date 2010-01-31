librule(name="allnn",
    headers=["allnn.h"],
    tests=["allnn_test.cc"],
    deplibs=["fastlib:fastlib", "fastlib/tree:tree" ]
    );
binrule(name="main",
    sources=["main.cc"],
    deplibs=[":allnn"]);
