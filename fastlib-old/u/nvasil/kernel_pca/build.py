librule(name="kpca",
    headers=["kernel_pca.h", "kernel_pca_impl.h"],
    deplibs=["sparse:sparse", "fastlib:fastlib", "la:la", 
    "u/nvasil/allknn:allknn"]
    );

binrule(name="kptest",
        sources=["kernel_pca_test.cc"],
        cflags=" -fexceptions", #required by trilinos
        deplibs=[":kpca"]
       );

