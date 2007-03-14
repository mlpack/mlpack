
librule(
    sources = ["textfile.cc"],
    headers = ["textfile.h"],
    deplibs = ["base:base", "col:col"],
    )

binrule(
    name = "textfile_test",
    sources = ["textfile_test.cc"],
    linkables = [":file", "fx:fx"]
    )
