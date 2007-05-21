
librule(name = "nbr",
   sources = ["blockdev.cc", "nbr_utils.cc", "cache.cc"],
   headers = ["blockdev.h", "dfs.h", "nbr_utils.h", "spbounds.h",
              "cache.h", "gnp.h", "kdtree.h", "spnode.h"],
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "tkde",
   sources = ["tkde.cc"],
   deplibs = [":nbr"])

binrule(name = "allnn",
   sources = ["allnn.cc"],
   deplibs = [":nbr"])



binrule(name = "cache_test",
   sources = ["cache_test.cc"],
   headers = ["cache.h", "blockdev.h"],
   deplibs = ["fastlib:fastlib_int"])


