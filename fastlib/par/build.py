
librule(
   sources = ["thread.cc"],
   headers = ["thread.h", "task.h", "grain.h"],
   deplibs = ["base:base", "col:col"])

librule(
   name = "mpi",
   sources = [],
   headers = ["mpigrain.h"],
   deplibs = [":par"])

