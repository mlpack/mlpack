librule(
	name = "metrics",
	sources = ["metrics.cc"],
	headers = ["metrics.h","globals.h"],
	deplibs = ["fastlib:fastlib"]
)

librule(
	name = "matcher",
	sources = ["matcher.cc"],
	headers = ["matcher.h","globals.h"],
	deplibs = ["fastlib:fastlib",":metrics"]
)

librule(
	name = "multi_matcher",
	sources = ["multi_matcher.cc"],
	headers = ["multi_matcher.h","globals.h"],
	deplibs = ["fastlib:fastlib",":matcher",":metrics"]
)

binrule(
	name = "naive_test",
	sources = ["naive_test.cc"],
	headers = ["globals.h"],
	deplibs = ["fastlib:fastlib",":metrics",":matcher"]
)
