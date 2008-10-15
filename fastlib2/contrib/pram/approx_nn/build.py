librule(name="approxnn",
    headers=["approx_nn.h"],
    #tests=["allnn_test.cc"],
    deplibs=["fastlib:fastlib", "fastlib/tree:tree" ]
    );
binrule(name="main",
    sources=["main.cc"],
    deplibs=[":approxnn"]);
