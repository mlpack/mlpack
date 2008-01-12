binrule(name = "fastica",
   sources = ["fastica.cc"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "linalg",
   sources = ["lin_alg.cc"],
   deplibs = ["fastlib:fastlib"])
