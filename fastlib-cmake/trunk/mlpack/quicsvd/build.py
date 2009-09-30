librule(
	name = "quicsvd",
	headers = ["cosine_tree.h","quicsvd.h"],
        sources = ["cosine_tree.cc", "quicsvd.cc"],
      	deplibs = ["fastlib:fastlib"],
	#tests = ["quic_svd_test.cc"]
)

binrule(
	# The name of the executable.
	name = "quicsvd_test",
	
	# The .c or .cc file containing main and any others you need.
	sources = ["quicsvd_test.cc"],
	
	# This line can be omitted if there are no headers.
	# headers = ["util.h","quic_svd_base.h","cosine_tree.h"],
	
	# The leading colon means to check this build.py for allnn.
	deplibs = [":quicsvd"]
)

binrule(
	# The name of the executable.
	name = "quicsvd_main",
	
	# The .c or .cc file containing main and any others you need.
	sources = ["quicsvd_main.cc"],
	
	# This line can be omitted if there are no headers.
	# headers = ["util.h","quic_svd_base.h","cosine_tree.h"],
	
	# The leading colon means to check this build.py for allnn.
	deplibs = [":quicsvd"]
)

binrule(
	# The name of the executable.
	name = "gen_kernel_matrix",
	
	# The .c or .cc file containing main and any others you need.
	sources = ["kernel_matrix_generator.cc"],
	
	# This line can be omitted if there are no headers.
	# headers = ["util.h","quic_svd_base.h","cosine_tree.h"],
	
	# The leading colon means to check this build.py for allnn.
	deplibs = [":quicsvd"]
)
