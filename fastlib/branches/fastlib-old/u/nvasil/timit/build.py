binrule(
		name="timit_nn",
    sources=["timit_nn.cc"],
		deplibs = ["fastlib:fastlib", "u/nvasil/mmanager:mmapmm",
							 "u/nvasil/tree:binarytree"]
		)

