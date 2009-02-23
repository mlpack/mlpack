librule(
	name="eri",
	headers=["eri.h", "shell_pair.h", "basis_shell.h", "basis_function.h"],
	sources=["eri.cc", "shell_pair.cc"],
	deplibs=["fastlib:fastlib"]
)

