librule(name="allkfn",
    headers=["allkfn.h"],
    tests=["allkfn_test.cc"],
    deplibs=["fastlib:fastlib"]
    );

binrule(name="allkfn_exe", 
    sources=["main.cc"],
    deplibs=["fastlib:fastlib", ":allkfn"])
