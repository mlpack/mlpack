librule(
    name="hk_convex_ise_lib",
    sources=[],
    headers=["hyperkernels.h","hk_convex_ise.h","ichol.h","dte.h","engine.h",
             "special_linear_algebra.h"],
    deplibs=["fastlib:fastlib_int"]
    )


binrule(
    name="hk_convex_ise",
    sources=["hk_convex_ise_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":hk_convex_ise_lib"]
    )



