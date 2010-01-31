librule(
    sources = ["textfile.cc"],
    headers = ["textfile.h"],
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
    deplibs = [":file", "fx:fx"]
    )


