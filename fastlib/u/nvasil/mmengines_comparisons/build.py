binrule(
		name="memcomp",
    sources=["main.cc"],
		linkables = ["fastlib:fastlib", "u/nvasil/mmanager:mmapmm",
		             "u/nvasil/mmanager_with_tpie:tpiemm",
							   "u/nvasil/tree:binarytree"]
		)
