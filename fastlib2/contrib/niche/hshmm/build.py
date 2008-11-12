binrule(
	name = "hshmm",
	sources = ["main.cc"],
	headers = ["hmm.h","hshmm.h","distribution.h"],
        deplibs = ["fastlib:fastlib"]
)
