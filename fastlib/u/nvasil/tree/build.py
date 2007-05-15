librule(
  	name="common",
		headers=["../mmanager/memory_manager.h"]+
		        lglob("../loki/*.h")+
						["point.h", "computations_counter.h",
						 "euclidean_metric.h", "null_statistics.h",
						 "point_identity_discriminator.h"],
						 deplibs=["u/nvasil/mmanager_with_tpie:tpiemm"]
		);
binrule(
		name="hrect_test",
		sources=["hyper_rectangle_unit.cc"],
		headers=["hyper_rectangle.h", "hyper_rectangle_impl.h"],
		linkables=["fastlib:fastlib", ":common"],	
		);
binrule(
		name="node_test",
		sources=["node_unit.cc"],
		headers=["./node_impl.h", "./node.h", "../dataset/*.h",
		         "hyper_rectangle.h", "hyper_rectangle_impl.h"],
						 linkables=["fastlib:fastlib", ":common"]
		);
binrule(
		name="tree_test",
		sources=["binary_tree_unit.cc"],
		headers=["./node_impl.h", "./node.h", "../dataset/*.h",
		         "hyper_rectangle.h", "hyper_rectangle_impl.h",
						 "binary_tree.h", "binary_tree_impl.h", 
						 "kd_pivoter1.h"],
						 linkables=["fastlib:fastlib", ":common"]
		);


