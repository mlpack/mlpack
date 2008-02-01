librule(
	name = "hf",
	headers = ["hf.h"],
	#sources = ["hf.cc"],
	deplibs = ["fastlib:fastlib"],
	tests = ["hf_test.cc"]
)