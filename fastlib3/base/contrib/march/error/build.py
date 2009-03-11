librule(
	name="error_tester",
	headers=["naive_kernel_sum.h", "hybrid_error_analysis.h", "hybrid_error.h", "hybrid_error_stat.h"],
	deplibs=["fastlib:fastlib"],
)

binrule(
	name="error_tester_main",
	sources=["hybrid_error.cc"],
	deplibs=["fastlib:fastlib", ":error_tester"],
)

binrule(
	name="naive_sum_main",
	headers=["naive_kernel_sum.h"],
	sources=["naive_kernel_sum.cc"],
	deplibs=["fastlib:fastlib"],
)