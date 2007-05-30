binrule(
		name="memcomp1",
    sources=["main.cc"],
		deplibs = ["fastlib:fastlib", "u/nvasil/mmanager:mmapmm",
		           "u/nvasil/mmanager_with_tpie:tpiemm",
							 "u/nvasil/tree:binarytree"]
		)
binrule(
		name="memcomp2",
    sources=["main.cc"],
		deplibs = ["fastlib:fastlib", "u/nvasil/mmanager:mmapmm",
		            "u/nvasil/mmanager_with_tpie:tpiemm",
							  "u/nvasil/tree:binarytree"]
		)

binrule(
		name="memcomp3",
    sources=["main.cc"],
		deplibs = ["fastlib:fastlib", "u/nvasil/mmanager:mmapmm",
		            "u/nvasil/mmanager_with_tpie:tpiemm",
							  "u/nvasil/tree:binarytree"]
		)
