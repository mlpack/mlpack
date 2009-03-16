librule(
	name="fock_matrix_comparison",
	headers=["fock_matrix_comparison.h"],
	sources=["fock_matrix_comparison.cc"],
	deplibs=["fastlib:fastlib"],
)

binrule(
	name="naive_comparison",
	headers=["naive_fock_matrix.h"],
	sources=["naive_comparison.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/fock_impl:eri"]
)

binrule(
	name="fock_matrix_main",
	headers=["cfmm/cfmm_coulomb.h"],
	sources=["fock_matrix_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/cfmm:cfmm_coulomb", "contrib/march/fock_matrix/naive:naive_fock_matrix", "contrib/march/fock_matrix/multi_tree:multi_tree_fock", "contrib/march/fock_matrix/link:link_exchange"],
)
