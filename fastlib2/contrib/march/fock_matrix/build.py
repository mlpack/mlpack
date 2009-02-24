binrule(
	name="naive_comparison",
	headers=["naive_fock_matrix.h"],
	sources=["naive_comparison.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/fock_impl:eri"]
)

