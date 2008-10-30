librule(
	name = "gm",
	headers = ["gm.h","bipartie.h","factor_graph.h"],
        sources = ["gm.cc"],
      	deplibs = ["fastlib:fastlib"],
	#tests = ["quic_svd_test.cc"]
)

binrule(
	# The name of the executable.
	name = "gm_test",
	
	# The .c or .cc file containing main and any others you need.
	sources = ["gm_test.cc"],
	
	# This line can be omitted if there are no headers.
	# headers = ["util.h","quic_svd_base.h","cosine_tree.h"],
	
	# The leading colon means to check this build.py for allnn.
	deplibs = [":gm"]
)

#binrule(
	# The name of the executable.
	#name = "quic_svd_main",
	
	# The .c or .cc file containing main and any others you need.
	#sources = ["quic_svd_main.cc"],
	
	# This line can be omitted if there are no headers.
	#headers = ["allnn_main.h"],
	
	# The leading colon means to check this build.py for allnn.
	#deplibs = [":quic_svd"]
#)
