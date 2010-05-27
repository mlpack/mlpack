
librule(name="approxnn",
    headers=["approx_nn.h"],
    deplibs=["fastlib:fastlib",
             "fastlib/tree:tree"]);
librule(name="approxnn_dual",
    headers=["approx_nn_dual.h"],
    deplibs=["fastlib:fastlib",
             "fastlib/tree:tree"]);
binrule(name="main",
    sources=["main.cc"],
    deplibs=[":approxnn",
             "mlpack/allknn:allknn"]);
binrule(name="main_dual",
    sources=["main_dual.cc"],
    deplibs=[":approxnn_dual",
             #"mlpack/allknn:allknn"
             ]);

