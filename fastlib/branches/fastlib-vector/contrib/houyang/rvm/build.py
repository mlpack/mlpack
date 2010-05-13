librule(
	name = "rvm",
	
	sources = ["rvm_main.cc"],
	
	headers = ["rvm.h"],
	
	deplibs = ["fastlib:fastlib"],
	
)

binrule(
    # The name of the executable.
    name = "rvm_main",
    
    sources = ["rvm_main.cc"],
   
    headers = ["rvm.h"],

    deplibs = [":rvm"]

    )

