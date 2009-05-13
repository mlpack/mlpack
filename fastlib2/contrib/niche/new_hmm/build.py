binrule(
    name = "zhmm",
    sources = ["zhmm.cc"],
    headers = ["zhmm.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "tc",
    sources = ["test_template_classes.cc"],
    headers = ["test_template_classes.h","multinomial.h","gaussian.h","mixture.h"],
    deplibs = ["fastlib:fastlib"]
    )
