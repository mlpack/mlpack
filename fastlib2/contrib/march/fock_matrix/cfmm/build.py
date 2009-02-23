binrule(
	name="cfmm_screening_main",
	headers=["../fock_impl/eri.h", "../../../dongryel/fast_multipole_method/continuous_fmm.h"],
	sources=["cfmm_screening_main.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri", "contrib/dongryel/multitree_template:multitree_template",
		 "contrib/dongryel/proximity_project:proximity_project",
		 "mlpack/series_expansion:series_expansion"]
)
