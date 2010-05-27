
binrule(name = "nbc-single",
   sources = ["nbc-single.cc"],
   deplibs = ["fastlib:fastlib_int", "fastlib/thor:thor"])

binrule(name = "nbc-multi",
   sources = ["nbc-multi.cc"],
   deplibs = ["fastlib:fastlib_int", "fastlib/thor:thor"])
