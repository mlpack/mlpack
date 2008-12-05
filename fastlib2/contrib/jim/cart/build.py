
librule(
    name = "cart_stuff",                  
    headers = ["cartree.h", "split_tree.h", "training_set.h"],    
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

binrule(
    name = "cart",
    sources = ["cart_driver.cc"],  
    deplibs = [":cart_stuff"]
    )
