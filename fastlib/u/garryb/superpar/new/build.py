
binrule(name = "tkde",
   sources = ["tkde.cc"],
   headers = ["spnode.h", "spbounds.h", "kdtree.h", "gnp.h"],
   linkables = ["fastlib:fastlib_int"])



binrule(name = "allnn",
   sources = ["allnn.cc"],
   headers = ["spnode.h", "spbounds.h", "kdtree.h", "gnp.h"],
   linkables = ["fastlib:fastlib_int"])


