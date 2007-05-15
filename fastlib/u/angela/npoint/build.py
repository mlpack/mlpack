librule(
	name = "metrics",
	sources = ["metrics.cc"],
	headers = ["metrics.h"],
	deplibs = ["fastlib:fastlib"]
)

librule(
	name = "matcher",
	sources = ["matcher.cc"],
	headers = ["matcher.h"],
	deplibs = ["fastlib:fastlib",":metrics"]
)

librule(
	name = "multi_matcher",
	sources = ["multi_matcher.cc"],
	headers = ["multi_matcher.h"],
	deplibs = ["fastlib:fastlib",":matcher",":metrics"]
)

binrule(
	name = "naive_test",
	sources = ["naive_test.cc"],
	headers = [],
	deplibs = ["fastlib:fastlib",":metrics",":matcher"]
)
