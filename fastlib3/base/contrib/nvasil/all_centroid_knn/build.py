librule(name="allcent",
    headers=["all_centroid_knn.h"],
    tests=["all_centroid_knn_test.cc"],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn"]
    );

binrule(name="test",
		sources=["all_centroid_knn_test.cc"],
    deplibs=["fastlib:fastlib", ":allcent"])
