binrule(
	name = "emst",
	sources = ["emst.cc"],
	headers = ["emst.h"],
	linkables = ["fastlib:fastlib"]
)

binrule(
	name = "union_find_test",
	sources = ["union_find_test.cc"],
	headers = ["union_find.h"],
	linkables = ["fastlib:fastlib"]
)

binrule(
	name = "testing",
	sources = ["testing.cc"],
	linkables = ["fastlib:fastlib"]
)