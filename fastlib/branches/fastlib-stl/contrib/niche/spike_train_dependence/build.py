binrule(
    name = "hs_spike",
    sources = ["hs_spike.cc", "hsic.cc"],
    headers = ["hs_spike.h", "hsic.h"],
    deplibs = ["fastlib:fastlib"]
    )
