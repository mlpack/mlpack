librule(
	name="fock_matrix_comparison",
	headers=["fock_matrix_comparison.h"],
	sources=["fock_matrix_comparison.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
)

binrule(
	name="naive_comparison",
	headers=["naive_fock_matrix.h"],
	sources=["naive_comparison.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/fock_impl:eri"]
)

binrule(
	name="old_fock_matrix_main",
	headers=["cfmm/cfmm_coulomb.h"],
	sources=["fock_matrix_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/cfmm:cfmm_coulomb", "contrib/march/fock_matrix/naive:naive_fock_matrix", "contrib/march/fock_matrix/multi_tree:multi_tree_fock", "contrib/march/fock_matrix/link:link_exchange", ":fock_matrix_comparison", "contrib/march/fock_matrix/chem_reader:chem_reader"],
)

binrule(
	name="fock_matrix_main",
	sources=["fock_matrix_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/naive:naive_fock_matrix", ":fock_matrix_comparison", "contrib/march/fock_matrix/chem_reader:chem_reader", "contrib/march/fock_matrix/link:link_exchange", "contrib/march/fock_matrix/cfmm:cfmm_coulomb", "contrib/march/fock_matrix/multi_tree:multi_tree_fock"],
        #cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
        cflags = "-L../libint/lib -lint"
)
