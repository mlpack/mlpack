
librule(
    sources = ["discrete.cc", "geometry.cc"],
    headers = ["discrete.h", "geometry.h", "kernel.h", "math.h"],
    deplibs = ["fastlib/base:base", "fastlib/col:col"]
    )

binrule(
    name = "math_test",
    sources = ["math_test.cc"],
    deplibs = [":math"])

