librule(name="nmflib",
		headers=["sdp_objectives.h", "sdp_objectives_impl.h", 
              "sdp_nmf_engine.h", "geometric_nmf.h",
              "geometric_nmf_impl.h", "geometric_nmf_impl.h",
              "geometric_nmf_engine.h","geometric_nmf_seq_engine.h"],
    deplibs=["fastlib:fastlib", "mlpack/allknn:allknn", "contrib/nvasil/l_bfgs:l_bfgs"] )

librule(name="gopnmflib",
    headers=["gop_nmf.h", "gop_nmf_impl.h"],
    tests=["gop_test.cc"],
    deplibs=["fastlib:fastlib",":nmflib", "contrib/nvasil/l_bfgs:l_bfgs" ]);

binrule(name="nmf",
		sources=["main.cc"],
		deplibs=["fastlib:fastlib", ":nmflib"])
