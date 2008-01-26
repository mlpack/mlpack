librule(
    sources = ["textfile.cc"],
    headers = ["textfile.h"],
    deplibs = ["fastlib/base:base", "fastlib/col:col"],
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
    deplibs = [":file", "fastlib/fx:fx"]
    )


