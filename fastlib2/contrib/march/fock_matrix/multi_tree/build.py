librule(
	name="multi_tree_fock",
	headers=["multi_tree_fock.h", "square_fock_tree.h"],
	sources=["multi_tree_fock.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
)

binrule(
    name="shell_tree_test",
    headers=["basis_shell_tree.h", "shell_tree_impl.h"],
    sources=["basis_shell_tree.cc", "shell_tree_impl.cc", "shell_tree_test.cc"],
    deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
    cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)
