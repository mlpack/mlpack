librule(name="mvu",
		headers=["mvu_objectives.h", "mvu_objectives_impl.h"],
		deplibs=["fastlib:fastlib"] )
binrule(name="ncmvu",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":mvu", "contrib/nvasil/l_bfgs:l_bfgs"])
