binrule(
	name = "test_hmm",
	sources = ["test_hmm.cc"],
	headers = ["hmm.h", "gaussian.h","mmk.h"],
        deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_lds",
    sources = ["test_lds.cc"],
    headers = ["lds.h", "gaussian.h", "mmk.h"],
    deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_gaussian",
    sources = ["test_gaussian.cc"],
    headers = ["gaussian.h", "mmk.h"],
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
