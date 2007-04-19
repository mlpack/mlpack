librule(
    sources = ["textfile.cc", "serialize.cc"],
    headers = ["textfile.h", "serialize.h"],
    deplibs = ["base:base", "col:col"],
    )

librule(
    name = "file_int",
    sources = [],
    headers = [],
    deplibs = [":file"],
    )

binrule(
    name = "textfile_test",
    sources = ["textfile_test.cc"],
    linkables = [":file", "fx:fx"]
    )


