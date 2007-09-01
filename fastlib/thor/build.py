
headers = """
blockdev.h
cachearray.h
cachearray_impl.h
cache.h
dfs.h
dfs_impl.h
distribcache.h
gnp.h
kdtree.h
kdtree_impl.h
rpc_base.h
rpc.h
rpc_sock.h
thor.h
thor_struct.h
thortree.h
thortree_impl.h
thor_utils.h
thor_utils_impl.h
work.h
work_impl.h
""".split()

librule(name = "thor",
   sources = ["distribcache.cc", "blockdev.cc", "work.cc",
              "rpc.cc", "rpc_sock.cc"],
   headers = headers,
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "allnn", sources = ["allnn.cc"], deplibs = [":thor"])
