librule(
	name = "nnsvm",	
	sources = ["nnsvm_main.cc"],	
	headers = ["nnsmo.h", "nnsvm.h"],	
	deplibs = ["fastlib:fastlib"],	
)

binrule(
   name = "nnsvm_bin",
   sources = ["nnsvm.cc"],
   headers = ["nnsmo.h", "nnsvm.h"],
   linkables = ["fastlib:fastlib"])

