
binrule(name = "nbc-single",
   sources = ["nbc-single.cc"],
   deplibs = ["fastlib:fastlib_int", "fastlib/thor:thor"])

binrule(name = "nbc-multi",
   sources = ["nbc-multi.cc"],
   deplibs = ["fastlib:fastlib_int", "fastlib/thor:thor"])

#The binary rule for findknn
binrule(
    name = "findknn",                     # the executable name
    sources = ["findknn.cc"],                   
    headers = ["findknn.h"], # no extra headers
    deplibs = ["fastlib:fastlib_int"]
    )

