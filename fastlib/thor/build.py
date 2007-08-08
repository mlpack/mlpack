
librule(name = "thor",
   sources = ["distribcache.cc", "blockdev.cc", "work.cc",
              "rpc.cc", "rpc_sock.cc"],
   headers = ["blockdev.h", "cachearray.h",
              "dfs.h", "gnp.h", "kdtree.h", "thor_utils.h",
              "thortree.h", "work.h",
              "rpc.h", "rpc_sock.h", "cache.h", "distribcache.h",
              "thor.h"],
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "allnn", sources = ["allnn.cc"], deplibs = [":thor"])
