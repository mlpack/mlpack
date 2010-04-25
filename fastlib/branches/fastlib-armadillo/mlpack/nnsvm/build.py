librule(
	name = "nnsvm",	

	sources = ["nnsvm_main.cc"],	

	headers = ["nnsmo.h", "nnsvm.h"],	

	deplibs = ["fastlib:fastlib"],	
)

binrule(
   name = "nnsvm_main",

   sources = ["nnsvm_main.cc"],

   headers = ["nnsmo.h", "nnsvm.h"],

   deplibs = [":nnsvm"]
)


