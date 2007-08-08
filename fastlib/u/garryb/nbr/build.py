
binrule(name = "tkde",
   sources = ["tkde.cc"],
   deplibs = ["thor:thor"])

binrule(name = "allnn_rpc",
   headers = ["allnn.cc"],
   sources = ["allnn_rpc.cc"],
   deplibs = ["thor:thor"])

binrule(name = "allnn",
   sources = ["allnn.cc"],
   deplibs = ["thor:thor"])

binrule(name = "range",
   sources = ["range.cc"],
   deplibs = ["thor:thor"])

binrule(name = "gravity",
   sources = ["gravity.cc"],
   deplibs = ["thor:thor"])

binrule(name = "affinity",
   sources = ["affinity.cc"],
   deplibs = ["thor:thor"])

binrule(name = "apcluster",
   sources = ["apcluster.c"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "makesim",
   sources = ["makesim.cc"],
   deplibs = ["fastlib:fastlib"])
