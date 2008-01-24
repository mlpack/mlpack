
librule(
    name = "md",                  
    headers = ["atom_tree.h", "lennard_jones.h"],     
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

librule(
    name = "md_test",                  
    headers = ["atom_tree.h", "lennard_jones.h", "test_lennard_jones.h"],     
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

binrule(
    name = "main",
    sources = ["LennardJones_main.cc"],
    linkables = [":md"]
    )

binrule(
    name = "test",
    sources = ["test_lennard_jones.cc"],  
    linkables = [":md_test"]
    )
