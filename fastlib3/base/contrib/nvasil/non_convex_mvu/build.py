librule(name="non_convex_mvu",
    headers=["non_convex_mvu.h", "non_convex_mvu_impl.h"],
    tests=["non_convex_mvu_test.cc"],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn", 
    "contrib/nvasil/allkfn:allkfn"])

binrule(name="ncmvu",
    sources=["main.cc"],
    deplibs=[":non_convex_mvu", "fastlib:fastlib"]);
    
