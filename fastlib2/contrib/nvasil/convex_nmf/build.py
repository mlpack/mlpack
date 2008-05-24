librule(name="nmflib",
		headers=["sdp_objectives.h", "sdp_objectives_impl.h", "sdp_nmf_engine.h"],
		deplibs=["fastlib:fastlib"] )
binrule(name="nmf",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":nmflib", "contrib/nvasil/l_bfgs:l_bfgs"])
