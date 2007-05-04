
librule(
    headers = ["fastlib.h"],
    deplibs = ["la:la", "base:base",
             "fx:fx", "file:file", "col:col",
             "data:data", "math:math"
              #, "tree:tree", "par:par"
             ]
    )

librule(
    name = "fastlib_int",
    headers = ["fastlib.h"],
    deplibs = ["la:la", "base:base",
             "fx:fx", "file:file_int", "col:col",
             "data:data", "math:math",
             "tree:tree", "par:par"
             ]
    )

binrule(
    name = "otrav_test",
    sources = ["otrav_test.cc"],
    linkables = [":fastlib_int"])
