
librule(
    sources = ["textfile.cc"],
    headers = ["textfile.h"],
    deplibs = ["base:base", "col:col"],
    )

librule(
    name = "file_int",
    sources = ["serialize.cc"],
    headers = ["serialize.h"],
    deplibs = [":file"],
    )

binrule(
    name = "textfile_test",
    sources = ["textfile_test.cc"],
    linkables = [":file", "fx:fx"]
    )
