binrule(
    name = "test_hmm_multinomial",
    sources = ["test_hmm_multinomial.cc", "load_profile.cc"],
    headers = ["hmm.h", "lds.h", "multinomial.h", "gaussian.h", "mmk.h","svm.h","svm.cc"],
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

binrule(
    name = "hmm_testing",
    sources = ["hmm_testing.cc","test_hmm_multinomial.cc","../mmf/mmf3.cc"],
    headers = ["hmm_testing.h","../svm/svm.h","../svm/smo.h"],
    deplibs = ["fastlib:fastlib","contrib/niche/mmf:hmm"]
    )

binrule(
    name = "hmm_generative_classifier",
    sources = ["hmm_generative_classifier.cc", "test_hmm_multinomial.cc","../mmf/mmf3.cc"],
    headers = ["hmm_testing.h","../svm/svm.h"],
    deplibs = ["fastlib:fastlib","contrib/niche/mmf:hmm"]
    )

binrule(
    name = "read_variable_file",
    sources = ["read_variable_file.cc", "test_hmm_multinomial.cc","../mmf/mmf3.cc"],
    headers = ["hmm_testing.h","../svm/svm.h"],
    deplibs = ["fastlib:fastlib","contrib/niche/mmf:hmm"]
    )

binrule(
    name = "perm_dna",
    sources = ["perm_dna.cc", "test_hmm_multinomial.cc","../mmf/mmf3.cc"],
    headers = ["hmm_testing.h","../svm/svm.h"],
    deplibs = ["fastlib:fastlib","contrib/niche/mmf:hmm"]
    )



binrule(
    name = "test_serial",
    sources = ["test_serial.cc"],
    headers = [],
    deplibs = ["fastlib:fastlib"]
    )


binrule(
    name = "test_obs_kernel",
    sources = ["test_obs_kernel.cc"],
    headers = ["mmk.h","ppk.h"],
    deplibs = ["fastlib:fastlib"]
    )
