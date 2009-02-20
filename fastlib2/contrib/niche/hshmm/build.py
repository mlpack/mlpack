binrule(
    name = "test_hmm_multinomial",
    sources = ["test_hmm_multinomial.cc", "load_profile.cc"],
    headers = ["hmm.h", "lds.h", "multinomial.h", "gaussian.h", "mmk.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_hmm_gaussian",
    sources = ["test_hmm_gaussian.cc"],
    headers = ["hmm.h", "lds.h", "multinomial.h", "gaussian.h", "mmk.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_lds",
    sources = ["test_lds.cc"],
    headers = ["hmm.h", "lds.h", "multinomial.h", "gaussian.h", "mmk.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_gaussian",
    sources = ["test_gaussian.cc"],
    headers = ["hmm.h", "lds.h", "multinomial.h", "gaussian.h", "mmk.h"],
    deplibs = ["fastlib:fastlib"]
)


binrule(
    name = "test_distribution",
    sources = ["test_distribution.cc"],
    headers = ["distribution.h", "multinomial.h", "gaussian.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_debug",
    sources = ["testdebug.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "load_profile",
    sources = ["load_profile.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
)
