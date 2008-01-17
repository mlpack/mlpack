binrule(
	name = "emst",
	sources = ["emst.cc"],
	headers = ["emst.h", "union_find.h", "dtb.h", "naive_boruvka.h"],
	deplibs = ["fastlib:fastlib"]
)

binrule(
	name = "union_find_test",
	sources = ["union_find_test.cc"],
	headers = ["union_find.h"],
	deplibs = ["fastlib:fastlib"]
)

binrule(
	name = "testing",
	sources = ["testing.cc"],
	deplibs = ["fastlib:fastlib"]
)