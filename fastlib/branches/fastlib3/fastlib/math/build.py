
librule(
    sources = ["discrete.cc", "geometry.cc", "statistics.cc"],
    headers = ["discrete.h",
               "geometry.h",
               "statistics.h",
               "kernel.h",
               "math_lib.h",
               "math_lib_impl.h"],
    deplibs = ["fastlib/base:base", "fastlib/col:col"]
    )

binrule(
    name = "math_test",
    sources = ["math_test.cc"],
    deplibs = [":math"])
