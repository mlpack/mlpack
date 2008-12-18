librule(name="optimization",
		headers=["optimization.h"],
		sources=["optimization.cc"],
		deplibs=["fastlib:fastlib"]);
		
librule(name="test_obj",
        headers=["test_obj.h"],
        sources=["test_obj.cc"],
        deplibs=["fastlib:fastlib"])
		
binrule(name="test_main", 
        sources=["test_main.cc"], 
        deplibs=[":optimization", ":test_obj"])

