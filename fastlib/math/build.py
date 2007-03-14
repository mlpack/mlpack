
librule(
    sources = ["discrete.cc", "geometry.cc"],
    headers = ["discrete.h", "geometry.h", "kernel.h", "math.h"],
    deplibs = ["base:base", "col:col"]
    )

binrule(
    name = "math_test",
    sources = ["math_test.cc"],
    linkables = [":math"])

