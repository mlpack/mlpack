librule(
		name = "datapack",
		sources = ["datapack.cc"],
		headers = ["datapack.h","globals.h"],
		deplibs = ["fastlib:fastlib"]
)

librule(
	name = "metrics",
	sources = ["metrics.cc"],
	headers = ["metrics.h","globals.h"],
	deplibs = ["fastlib:fastlib",":datapack"]
)

librule(
	name = "matcher",
	sources = ["matcher.cc"],
	headers = ["matcher.h","globals.h"],
	deplibs = ["fastlib:fastlib",":datapack",":metrics"]
)

librule(
	name = "naive",
	sources = ["naive.cc"],
	headers = ["naive.h","globals.h"],
	deplibs = ["fastlib:fastlib",":datapack",":metrics",":matcher"]
)

librule(
	name = "multi_matcher",
	sources = ["multi_matcher.cc"],
	headers = ["multi_matcher.h","globals.h"],
	deplibs = ["fastlib:fastlib",":datapack",":metrics",":matcher"]
)

binrule(
	name = "npoint",
	sources = ["main.cc"],
	headers = ["globals.h"],
	deplibs = ["fastlib:fastlib",":datapack",":metrics",":matcher",":naive"]
)
