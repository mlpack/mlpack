librule(name="mvu",
		headers=["mvu_objectives.h", "mvu_objectives_impl.h"],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn", "mlpack/allkfn:allkfn"] )
binrule(name="ncmvu",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":mvu", "fastlib/optimization/lbfgs:lbfgs"])
