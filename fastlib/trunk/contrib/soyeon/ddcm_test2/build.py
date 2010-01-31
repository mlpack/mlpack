librule(name="sampling",
        headers=["sampling.h"],
        sources=["sampling.cc"],
        deplibs=["fastlib:fastlib"]);
        
librule(name="objective2",
		headers=["objective2.h"],
		sources=["objective2.cc"],
		deplibs=["fastlib:fastlib"]);

librule(name="objective3",
		headers=["objective3.h"],
		sources=["objective3.cc"],
		deplibs=["fastlib:fastlib"]);

		
librule(name="optimization",
		headers=["optimization.h"],
		sources=["optimization.cc"],
		deplibs=["fastlib:fastlib"]);
		
	
binrule(name="main", 
        sources=["ddcm_test.cc"], 
        deplibs=[":objective2", ":sampling", ":optimization"])
        
binrule(name="eval", 
sources=["test_eval2.cc"], 
deplibs=[":objective2", ":sampling", ":optimization"])


binrule(name="mainlog", 
        sources=["ddcm_test.cc"], 
        deplibs=[":objective3", ":sampling", ":optimization"])



librule(name="MLsampling",
        headers=["MLsampling.h"],
        sources=["MLsampling.cc"],
        deplibs=["fastlib:fastlib"]);
        
librule(name="MLobjective",
		headers=["MLobjective.h"],
		sources=["MLobjective.cc"],
		deplibs=["fastlib:fastlib"]);

binrule(name="MLmain", 
        sources=["ML_test.cc"], 
        deplibs=[":MLobjective", ":MLsampling", ":optimization"])

binrule(name="MLeval", 
sources=["ML_test_eval.cc"], 
deplibs=[":MLobjective", ":MLsampling", ":optimization"])



     
librule(name="MLPsampling",
        headers=["MLPsampling.h"],
        sources=["MLPsampling.cc"],
        deplibs=["fastlib:fastlib"]);
        
librule(name="MLPobjective",
		headers=["MLPobjective.h"],
		sources=["MLPobjective.cc"],
		deplibs=["fastlib:fastlib"]);

binrule(name="MLPmain", 
        sources=["MLP_test.cc"], 
        deplibs=[":MLPobjective", ":MLPsampling", ":optimization"])

binrule(name="MLPeval", 
sources=["MLP_test_eval.cc"], 
deplibs=[":MLPobjective", ":MLPsampling", ":optimization"])




librule(name="Dsampling",
        headers=["Dsampling.h"],
        sources=["Dsampling.cc"],
        deplibs=["fastlib:fastlib"]);
        
librule(name="Dobjective2",
		headers=["Dobjective2.h"],
		sources=["Dobjective2.cc"],
		deplibs=["fastlib:fastlib"]);

binrule(name="Dmain", 
        sources=["D_test.cc"], 
        deplibs=[":Dobjective2", ":Dsampling", ":optimization"])

binrule(name="Deval", 
sources=["D_test_eval.cc"], 
deplibs=[":Dobjective2", ":Dsampling", ":optimization"])



     
