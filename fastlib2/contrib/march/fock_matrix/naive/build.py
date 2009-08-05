librule (
	name="naive_fock_matrix",
	headers=["naive_fock_matrix.h"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"]
)

binrule(
	name="naive_fock_main",
	headers=["naive_fock_matrix.h"],
	sources=["naive_fock_main.cc"],
	deplibs=["fastlib:fastlib", ":naive_fock_matrix", "contrib/march/fock_matrix/chem_reader:chem_reader", "contrib/march/fock_matrix/fock_impl:eri"],
        cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)
