
librule(name = "nbr",
   sources = ["blockdev.cc", "nbr_utils.cc",
              "cache.cc", "work.cc", "rpc_sock.cc"],
   headers = ["blockdev.h", "cache.h", "cachearray.h",
              "dfs.h", "gnp.h", "kdtree.h", "nbr_utils.h",
              "spbounds.h", "spnode.h", "work.h", "rpc_sock.h"],
   deplibs = ["fastlib:fastlib_int"])

librule(name = "nbr_mpi",
   sources = ["rpc.cc", "netcache.cc"],
   headers = ["rpc.h", "netcache.h"],
   deplibs = [":nbr"])

binrule(name = "tkde",
   sources = ["tkde.cc"],
   deplibs = [":nbr"])

binrule(name = "tkde_mpi",
   sources = ["tkde.cc"],
   deplibs = [":nbr_mpi"])

binrule(name = "allnn_mpi",
   sources = ["allnn.cc"],
   deplibs = [":nbr_mpi"])

binrule(name = "allnn",
   sources = ["allnn.cc"],
   deplibs = [":nbr"])

binrule(name = "gravity",
   sources = ["gravity.cc"],
   deplibs = [":nbr"])

binrule(name = "tpc",
   sources = ["tpc.cc"],
   deplibs = [":nbr"])

binrule(name = "affinity",
   sources = ["affinity.cc"],
   deplibs = [":nbr"])

binrule(name = "affinity2",
   sources = ["affinity2.cc"],
   deplibs = [":nbr"])

binrule(name = "apcluster",
   sources = ["apcluster.c"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "makesim",
   sources = ["makesim.cc"],
   deplibs = ["fastlib:fastlib"])

binrule(name = "cache_test",
   sources = ["cache_test.cc"],
   headers = ["cache.h", "blockdev.h"],
   deplibs = ["fastlib:fastlib_int"])


