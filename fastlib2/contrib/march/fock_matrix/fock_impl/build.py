librule(
    name="libint_wrappers",
    headers=["libint_wrappers.h"],
    sources=["libint_wrappers.cc"],
    deplibs=["fastlib:fastlib"],
    cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)

librule(
	name="eri",
	headers=["eri.h", "shell_pair.h", "basis_shell.h", "basis_function.h"],
	sources=["eri.cc", "shell_pair.cc"],
	deplibs=["fastlib:fastlib"],
)

librule(
	name="oeints",
	headers=["oeints.h", "basis_shell.h"],
	sources=["oeints.cc"],
	deplibs=["fastlib:fastlib", ":eri"]
)
