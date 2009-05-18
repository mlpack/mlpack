binrule(
    name = "zhmm",
    sources = ["zhmm.cc"],
    headers = ["zhmm.h","multinomial.h","diag_gaussian.h","mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "tc",
    sources = ["test_template_classes.cc"],
    headers = ["test_template_classes.h","multinomial.h","gaussian.h","mixture.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "testMatrix",
    sources = ["testMatrix.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )
