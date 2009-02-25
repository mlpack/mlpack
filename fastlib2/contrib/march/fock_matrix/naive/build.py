librule (
	name="naive_fock_matrix",
	headers=["naive_fock_matrix.h"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"]
)

binrule(
	name="naive_fock_matrix_main",
	headers=["naive_fock_matrix.h"],
	sources=["naive_fock_matrix_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"]
)