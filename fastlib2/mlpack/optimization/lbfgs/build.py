librule(name="l_bfgs",
    headers=["l_bfgs.h", "l_bfgs_impl.h", "optimization_utils.h"],
    deplibs=["fastlib:fastlib"])

binrule(name="test",
    sources=["test_l_bfgs.cc"],
    deplibs=["fastlib:fastlib", ":l_bfgs", "mlpack/allknn:allknn", 
    "contrib/nvasil/allkfn:allkfn", "contrib/nvasil/mvu:mvu"])

