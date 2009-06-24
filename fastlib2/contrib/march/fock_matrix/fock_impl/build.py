librule(
	name="eri",
	headers=["eri.h", "shell_pair.h", "basis_shell.h", "basis_function.h", "libint_wrappers.h"],
	sources=["eri.cc", "shell_pair.cc", "libint_wrappers.cc"],
	deplibs=["fastlib:fastlib"],
        cflags = "-L../../libint/lib/ -lint"
)

librule(
	name="oeints",
	headers=["oeints.h", "basis_shell.h"],
	sources=["oeints.cc"],
	deplibs=["fastlib:fastlib", ":eri"]
)
