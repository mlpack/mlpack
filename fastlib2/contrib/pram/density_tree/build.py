
librule(
    name = "dtree",                  
    headers = ["dtree.h"],    
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "main",
    sources = ["main.cc"],  
    deplibs = [":dtree"]
    )
