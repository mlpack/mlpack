librule(
	name="schwartz_prescreening",
	headers=["schwartz_prescreening.h"],
	sources=["schwartz_prescreening.cc"],
	deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:eri"],
	tests=[]
)

binrule(
	name="schwartz_prescreening_main",
        headers=[],
	sources=["schwartz_prescreening_main.cc"],
	deplibs=["fastlib:fastlib", ":schwartz_prescreening", "contrib/march/fock_matrix/fock_impl:eri"]
)

