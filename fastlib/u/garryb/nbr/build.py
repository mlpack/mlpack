
binrule(name = "tkde",
   sources = ["tkde.cc"],
   headers = ["spnode.h", "spbounds.h", "kdtree.h", "gnp.h", "dfs.h"],
   deplibs = ["fastlib:fastlib_int"])



binrule(name = "allnn",
   sources = ["allnn.cc"],
   headers = ["spnode.h", "spbounds.h", "kdtree.h", "gnp.h", "dfs.h"],
   deplibs = ["fastlib:fastlib_int"])



binrule(name = "cache_test",
   sources = ["cache_test.cc"],
   headers = ["cache.h", "blockdev.h"],
   deplibs = ["fastlib:fastlib_int"])


