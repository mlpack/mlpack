librule(name="l_bfgs",
    headers=["l_bfgs.h", "l_bfgs_impl.h"],
    tests=["test_l_bfgs.cc"],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn", 
    "contrib/nvasil/allkfn:allkfn"])


