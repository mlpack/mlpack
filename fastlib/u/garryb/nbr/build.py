
librule(name = "nbr",
   sources = ["blockdev.cc", "nbr_utils.cc", "cache.cc", "work.cc"],
   headers = ["blockdev.h", "dfs.h", "nbr_utils.h", "spbounds.h",
              "cache.h", "gnp.h", "kdtree.h", "spnode.h", "work.h"],
   deplibs = ["fastlib:fastlib_int"])

librule(name = "nbr_mpi",
   sources = ["blockdev.cc", "nbr_utils.cc",
              "cache.cc", "work.cc", "rpc.cc"],
   headers = ["blockdev.h", "dfs.h", "nbr_utils.h", "spbounds.h",
              "cache.h", "gnp.h", "kdtree.h", "spnode.h", "work.h",
              "rpc.h"],
   deplibs = ["fastlib:fastlib_int"],
   cflags = "-DUSE_MPI")

binrule(name = "tkde",
   sources = ["tkde.cc"],
   deplibs = [":nbr"])

binrule(name = "allnn_mpi",
   sources = ["allnn.cc"],
   deplibs = [":nbr_mpi"],
   cflags = "-DUSE_MPI")



binrule(name = "cache_test",
   sources = ["cache_test.cc"],
   headers = ["cache.h", "blockdev.h"],
   deplibs = ["fastlib:fastlib_int"])


