binrule(
	name = "emst",
	sources = ["emst.cc"],
	headers = ["emst.h", "emsttree.h"],
	linkables = ["fastlib:fastlib"]
)