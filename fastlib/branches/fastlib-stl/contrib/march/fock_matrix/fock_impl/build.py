librule(
	name="eri",
	headers=["eri.h", "shell_pair.h", "basis_shell.h", "integral_tensor.h"],
	sources=["eri.cc", "shell_pair.cc", "integral_tensor.cc"],
	deplibs=["fastlib:fastlib"],
        cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)

binrule(
    name="eri_test",
    sources=["eri_test.cc"],
    deplibs=[":eri"],
    cflags = "-L/Users/march/Desktop/fastlib2/contrib/march/libint/lib -lint"
)
