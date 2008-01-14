binrule(name = "fastica",
   sources = ["fastica_full.cc"],
   headers = ["lin_alg.h"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "linalg",
   sources = ["test_lin_alg.cc"],
   headers = ["lin_alg.h"],
   deplibs = ["fastlib:fastlib"])
