binrule(name = "fastica",
   sources = ["fastica_stylish.cc"],
   headers = ["lin_alg.h", "fastica_stylish.h"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "linalg",
   sources = ["test_lin_alg.cc"],
   headers = ["lin_alg.h"],
   deplibs = ["fastlib:fastlib"])
