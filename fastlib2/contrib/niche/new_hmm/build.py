binrule(
    name = "test_isotropic_gaussian",
    sources = ["test_isotropic_gaussian.cc"],
    headers = ["isotropic_gaussian.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "generative_mmk",
    sources = ["generative_mmk.cc"],
    headers = ["generative_mmk.h",
               "generative_mmk_impl.h",
               "isotropic_gaussian.h",
               "multinomial.h",               
               "hmm.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "test_latent_mmk",
    sources = ["test_latent_mmk.cc"],
    headers = ["latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "latent_mmk",
    sources = ["latent_mmk.cc"],
    headers = ["latent_mmk.h",
               "latent_mmk_impl.h",
               "hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "hmm",
    sources = ["hmm.cc"],
    headers = ["hmm.h",
               "multinomial.h",
               "diag_gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib",
               "contrib/niche/kmeans_nonempty:kmeans_nonempty"]
    )

binrule(
    name = "tc",
    sources = ["test_template_classes.cc"],
    headers = ["test_template_classes.h",
               "multinomial.h",
               "gaussian.h",
               "mixture.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "testMatrix",
    sources = ["testMatrix.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )
