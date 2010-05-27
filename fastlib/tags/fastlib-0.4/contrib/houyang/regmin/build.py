librule(
	name = "svm",
	
	sources = ["regmin.cc"],
	
        headers = ["opt_smo.h", "opt_sgd.h", "opt_md.h", "opt_tgd.h", "regmin.h", "regmin_data.h"],
	
	deplibs = ["fastlib:fastlib"],
	
)

binrule(
    # The name of the executable.
    name = "regmin",
    
    sources = ["regmin.cc"],
   
    headers = ["opt_smo.h", "opt_sgd.h", "opt_md.h", "opt_tgd.h", "regmin.h", "regmin_data.h"],

    deplibs = [":svm"]

    )

