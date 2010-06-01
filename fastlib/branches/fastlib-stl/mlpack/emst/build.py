librule(
	name = "union_find",
	headers = ["union_find.h"],
	deplibs = ["fastlib:fastlib"],
	tests = ["union_find_test.cc"]
)

librule(
	name = "dtb",
	headers = ["emst.h", "dtb.h"],
	deplibs = ["fastlib:fastlib", ":union_find"]
)

binrule(
	name = "emst_main",
	sources = ["emst_main.cc"],
	deplibs = ["fastlib:fastlib", ":union_find", ":dtb"]
)
