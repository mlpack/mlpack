librule(
	name="multi_tree_fock",
	headers=["multi_tree_fock.h", "square_fock_tree.h"],
	sources=["multi_tree_fock.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
)