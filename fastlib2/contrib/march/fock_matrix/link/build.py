librule(
	name="link_exchange",
	headers=["link.h", "../fock_impl/basis_shell.h", "../fock_impl/shell_pair.h"],
	sources=["link.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
	tests=[]
)


binrule(
	name="link_main",
	headers=["link.h"],
	sources=["link_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri", ":link_exchange"]
)	