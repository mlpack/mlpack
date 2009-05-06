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
		
	
binrule(name="main", 
        sources=["ddcm_test.cc"], 
        deplibs=[":objective2", ":sampling", ":optimization"])
        
binrule(name="eval", 
sources=["test_eval.cc"], 
deplibs=[":objective2", ":sampling", ":optimization"])



