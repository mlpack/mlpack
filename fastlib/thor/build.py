
headers = """
blockdev.h
cachearray.h
cache.h
dfs.h
distribcache.h
gnp.h
kdtree_builder.h
kdtree.h
rpc_base.h
rpc.h
rpc_sock.h
thor.h
thor_struct.h
thortree_algs.h
thortree.h
thor_utils.h
work.h
""".split()

librule(name = "thor",
   sources = ["distribcache.cc", "blockdev.cc", "work.cc",
              "rpc.cc", "rpc_sock.cc"],
   headers = headers,
   deplibs = ["fastlib:fastlib_int"])

binrule(name = "allnn", sources = ["allnn.cc"], deplibs = [":thor"])
