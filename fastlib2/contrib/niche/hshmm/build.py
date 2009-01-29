binrule(
	name = "hshmm",
	sources = ["main.cc"],
	headers = ["hmm.h","distribution.h","hmm_distance.h"],
        deplibs = ["fastlib:fastlib"]
)

binrule(
    name = "test_obs",
    sources = ["test_obs_kernel.cc"],
    headers = ["hmm.h","distribution.h","hmm_distance.h"],
    deplibs = ["fastlib:fastlib"]
)
