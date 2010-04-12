librule(name="mvu",
		headers=["mvu_objectives.h", "mvu_objectives_impl.h",
    "mvu_classification.h", "mvu_classification_impl.h",],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn", "contrib/nvasil/allkfn:allkfn"] )
binrule(name="ncmvu",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":mvu", "contrib/nvasil/l_bfgs:l_bfgs"])
binrule(name="ncmvuc",
		sources=["main_classifier.cc"],
		deplibs=["fastlib:fastlib", ":mvu", "contrib/nvasil/l_bfgs:l_bfgs"])
