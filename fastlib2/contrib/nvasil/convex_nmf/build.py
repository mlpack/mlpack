librule(name="nmflib",
		headers=["nmf_objectives.h", "nmf_objectives_impl.h", "nmf_engine.h"],
		deplibs=["fastlib:fastlib"] )
binrule(name="nmf",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":nmflib", "contrib/nvasil/l_bfgs:l_bfgs"])
