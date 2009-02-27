librule(
	name = "svm",
	
	sources = ["svm_main.cc"],
	
        headers = ["opt_smo.h", "opt_sgd.h", "svm.h"],
	
	deplibs = ["fastlib:fastlib"],
	
)

binrule(
    # The name of the executable.
    name = "svm_main",
    
    sources = ["svm_main.cc"],
   
    headers = ["opt_smo.h", "opt_sgd.h", "svm.h"],

    deplibs = [":svm"]

    )

