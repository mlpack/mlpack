
librule(name = "nbr",
   sources = ["distribcache.cc", "blockdev.cc", "nbr_utils.cc",
              "work.cc",
              "rpc.cc", "rpc_sock.cc"],
   headers = ["blockdev.h", "cachearray.h",
              "dfs.h", "gnp.h", "kdtree.h", "nbr_utils.h",
              "spbounds.h", "spnode.h", "work.h",
              "rpc.h", "rpc_sock.h", "distribcache.h",
              "cache.h"],
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "rpc_sock_test",
   sources = ["rpc_sock_test.cc"],
   deplibs = [":nbr"])

# No more MPI
#librule(name = "nbr_mpi",
#   sources = ["rpc.cc", "netcache.cc"],
#   headers = ["rpc.h", "netcache.h"],
#   deplibs = [":nbr"])

binrule(name = "tkde",
   sources = ["tkde.cc"],
   deplibs = [":nbr"])

#binrule(name = "tkde_mpi",
#   sources = ["tkde.cc"],
#   deplibs = [":nbr_mpi"])

binrule(name = "allnn_rpc",
   sources = ["allnn_rpc.cc"],
   deplibs = [":nbr"])

binrule(name = "allnn",
   sources = ["allnn.cc"],
   deplibs = [":nbr"])

#--------

binrule(name = "range_rpc",
   sources = ["range_rpc.cc"],
   deplibs = [":nbr"])

binrule(name = "range",
   sources = ["range.cc"],
   deplibs = [":nbr"])

#--------

binrule(name = "akde_rpc",
   sources = ["akde_rpc.cc"],
   deplibs = [":nbr"])

binrule(name = "akde",
   sources = ["akde.cc"],
   deplibs = [":nbr"])

#--------

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


