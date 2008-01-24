librule(name="kpca",
    headers=["kernel_pca.h", "kernel_pca_impl.h", "allknn.h"],
    deplibs=["sparse:sparse", "fastlib:fastlib", "la:la"]
    );

binrule(name="kptest",
        sources=["kernel_pca_test.cc"],
        cflags=" -fexceptions",
        deplibs=[":kpca"]
       );

