
librule(
    sources = ["col.cc"],
    headers = ["arraylist.h", "heap.h", "string.h", "fastalloc.h", "intmap.h"],
    deplibs = ["base:base"]
    )

binrule(
    name = "col_test",
    sources = ["col_test.cc"],
    linkables = [":col"])


binrule(
    name = "timing_test",
    sources = ["timing_test.cc"],
    linkables = [":col", "fx:fx"])

