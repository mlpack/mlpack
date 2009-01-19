binrule(
	name = "hshmm",
	sources = ["main.cc"],
	headers = ["hmm.h","distribution.h","hmm_distance.h"],
        deplibs = ["fastlib:fastlib"]
)
