librule(
  	name="binarytree",
		headers=lglob("*.h"),
		deplibs=["fastlib:fastlib",
		         "u/nvasil/loki:loki"
						 ]
		);

librule(name="fortests",
		deplibs=["u/nvasil/dataset:bindataset",
		         "u/nvasil/mmanager:mmapmm",
						 "u/nvasil/mmanager_with_tpie:tpiemm"])
binrule(
		name="hrect_test",
		sources=["hyper_rectangle_unit.cc"],
		deplibs=[":binarytree", ":fortests"],	
		);
binrule(
		name="node_test",
		sources=["node_unit.cc"],
		deplibs=[":binarytree", ":fortests"]
		);
binrule(
		name="tree_test",
		sources=["binary_tree_unit.cc"],
		deplibs=[":binarytree", ":fortests"]
		);


