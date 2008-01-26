
librule(
    headers = ["fastlib.h"],
    deplibs = ["fastlib/la:la", "fastlib/base:base",
             "fastlib/fx:fx", "fastlib/file:file", "fastlib/col:col",
             "fastlib/data:data", "fastlib/math:math"
              #, "tree:tree", "par:par"
             ]
    )

librule(
    name = "fastlib_int",
    headers = ["fastlib.h"],
    deplibs = ["fastlib/la:la", "fastlib/base:base",
             "fastlib/fx:fx", "fastlib/file:file_int", "fastlib/col:col",
             "fastlib/data:data", "fastlib/math:math",
             "fastlib/tree:tree", "fastlib/par:par"
             ]
    )

binrule(
    name = "otrav_test",
    sources = ["otrav_test.cc"],
    deplibs = [":fastlib_int"])
