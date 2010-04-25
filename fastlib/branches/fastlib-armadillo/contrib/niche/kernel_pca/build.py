librule(
	name = "kernel_pca",
	
	headers = ["kernel_pca.h"],
	
	deplibs = ["fastlib:fastlib"],
	
)



binrule(
	
	name = "kernel_pca_main_standalone",
	
	sources = ["kernel_pca_main.cc"],
	
	headers = ["kernel_pca.h"],
	
	deplibs = ["fastlib:fastlib"]
)


binrule(
	
	name = "kernel_pca_main_from_lib",
	
	sources = ["kernel_pca_main.cc"],
	
	deplibs = [":kernel_pca"]
)
