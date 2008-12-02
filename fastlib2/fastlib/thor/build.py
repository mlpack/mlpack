
headers = """
blockdev.h
cachearray.h
cachearray_impl.h
cache.h
dfs.h
dfs_impl.h
rbfs.h
rbfs_impl.h
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
sched.h
sched_impl.h
""".split()

librule(name = "thor",
   sources = ["distribcache.cc", "blockdev.cc", "sched.cc",
              "rpc.cc", "rpc_sock.cc"],
   headers = headers,
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "allnn", sources = ["allnn.cc"], deplibs = [":thor"])
binrule(name = "allnnbfs", sources = ["allnnbfs.cc"], deplibs = [":thor"])
