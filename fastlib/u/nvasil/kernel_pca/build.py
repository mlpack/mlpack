binrule(name="kptest",
        headers=["kernel_pca.h", "kernel_pca_impl.h", "allknn.h"],
        sources=["kernel_pca_test.cc"],
        cflags=" -fexceptions",
        deplibs=["sparse:sparse", "fastlib:fastlib", "la:la"]
       );

