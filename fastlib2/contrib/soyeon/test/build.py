binrule(name="hello",
        sources=["hello_world.cc"],
        deplibs=["fastlib:fastlib"])

librule(name="lin_test",
        headers=["linear_algebra_class.h"],
        sources=["linear_algebra_class.cc"],
        deplibs=["fastlib:fastlib"])

binrule(name="main", 
        sources=["main.cc"], 
        deplibs=[":lin_test"])
