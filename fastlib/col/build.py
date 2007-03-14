
librule(
    sources = ["col.cc"],
    headers = ["arraylist.h", "heap.h", "string.h"],
    deplibs = ["base:base"]
    )

binrule(
    name = "col_test",
    sources = ["col_test.cc"],
    linkables = [":col"])

