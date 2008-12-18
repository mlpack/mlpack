librule(name="sampling",
        headers=["sampling.h"],
        sources=["sampling.cc"],
        deplibs=["fastlib:fastlib"]);
        
librule(name="objective2",
		headers=["objective2.h"],
		sources=["objective2.cc"],
		deplibs=["fastlib:fastlib"]);
		
librule(name="optimization",
		headers=["optimization.h"],
		sources=["optimization.cc"],
		deplibs=["fastlib:fastlib"]);
		
librule(name="test_obj",
        headers=["test_obj.h"],
        sources=["test_obj.cc"],
        deplibs=["fastlib:fastlib"])
		
binrule(name="main", 
        sources=["DDCM_test.cc"], 
        deplibs=[":objective2", ":sampling", ":optimization", ":test_obj"])

	
binrule(name="test_main", 
        sources=["test_main.cc"], 
        deplibs=[":objective2", ":sampling", ":optimization", ":test_obj"])

