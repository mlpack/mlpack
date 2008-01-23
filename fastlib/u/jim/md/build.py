
librule(
    name = "md",                  
    headers = ["AtomTree.h", "LennardJones.h"],     
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

binrule(
    name = "main",
    sources = ["LennardJones_main.cc"],
    linkables = [":md"]
    )

