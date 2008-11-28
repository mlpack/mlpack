
librule(
    name = "cart",                  
    headers = ["cartree.h", "split_tree.h", "training_set.h"],    
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

binrule(
    name = "main",
    sources = ["cart_driver.cc"],  
    deplibs = [":cart"]
    )
