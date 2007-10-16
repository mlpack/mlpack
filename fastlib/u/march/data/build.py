binrule(
	name = "protein_conversion",
	sources = ["protein_conversion.cc"],
	headers = ["protein_conversion.h"],
	linkables = ["fastlib:fastlib"]

)