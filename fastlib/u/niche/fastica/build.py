librule(
	name = "fastica_lib",
	headers = ["fastica.h", "lin_alg.h"],
	deplibs = ["fastlib:fastlib"]
)

binrule(
	name = "fastica",
	sources = ["fastica_main.cc"],
	headers = ["lin_alg.h"],
	deplibs = [":fastica_lib"]
)

binrule(name = "linalg",
   sources = ["test_lin_alg.cc"],
   headers = ["lin_alg.h"],
   deplibs = ["fastlib:fastlib"])
