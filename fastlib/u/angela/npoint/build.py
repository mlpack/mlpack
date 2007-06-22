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
		name = "datapack",
		sources = ["datapack.cc"],
		headers = ["datapack.h","globals.h"],
		deplibs = ["fastlib:fastlib"]
)

librule(
	name = "naive",
	sources = ["naive.cc"],
	headers = ["naive.h","globals.h"],
	deplibs = ["fastlib:fastlib",":metrics",":matcher",":datapack"]
)

librule(
	name = "multi_matcher",
	sources = ["multi_matcher.cc"],
	headers = ["multi_matcher.h","globals.h"],
	deplibs = ["fastlib:fastlib",":metrics",":matcher"]
)

binrule(
	name = "npoint",
	sources = ["main.cc"],
	headers = ["globals.h"],
	deplibs = ["fastlib:fastlib",":metrics",":matcher",":naive",":datapack"]
)
