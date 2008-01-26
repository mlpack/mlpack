
librule(
    sources = ["col.cc"],
    headers = ["arraylist.h", "heap.h", "string.h",
               "fastalloc.h", "intmap.h", "rangeset.h", "queue.h"],
    deplibs = ["fastlib/base:base"]
    )

binrule(
    name = "col_test",
    sources = ["col_test.cc"],
    deplibs = [":col"])


binrule(
    name = "timing_test",
    sources = ["timing_test.cc"],
    deplibs = [":col", "fastlib/fx:fx"])

