librule(
	name="multi_tree_fock",
	headers=["multi_tree_fock.h", "basis_shell_tree.h", "shell_tree_impl.h", "matrix_tree_impl.h", "matrix_tree.h", "eri_bounds.h"],
	sources=["multi_tree_fock.cc", "basis_shell_tree.cc", "shell_tree_impl.cc", "matrix_tree.cc", "matrix_tree_impl.cc", "eri_bounds.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
)

binrule(
    name="shell_tree_test",
    headers=["basis_shell_tree.h", "shell_tree_impl.h"],
    sources=["basis_shell_tree.cc", "shell_tree_impl.cc", "shell_tree_test.cc"],
    deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
    cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)

binrule(
    name="matrix_tree_test",
    headers=["basis_shell_tree.h", "shell_tree_impl.h", "matrix_tree.h", "matrix_tree_impl.h"],
    sources=["basis_shell_tree.cc", "shell_tree_impl.cc", "matrix_tree.cc", "matrix_tree_impl.cc", "matrix_tree_test.cc"],
    deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri", "contrib/march/fock_matrix/chem_reader:chem_reader"],
    cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)
