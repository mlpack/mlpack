librule(name="kpca",
    headers=["kernel_pca.h", "kernel_pca_impl.h"],
    deplibs=["fastlib/sparse:sparse", "fastlib:fastlib", "fastlib/la:la", 
    "../allknn:allknn"]
    );

binrule(name="kptest",
        sources=["kernel_pca_test.cc"],
        cflags=" -fexceptions", #required by trilinos
        deplibs=[":kpca"]
       );

