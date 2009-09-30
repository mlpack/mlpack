librule(
    name = "nbc",
    #sources = [],
    headers = ["simple_nbc.h",
               "phi.h",
               "math_functions.h"],
    deplibs = ["fastlib:fastlib"],
    tests = ["test_simple_nbc_main.cc"]
    )

binrule(
    name = "nbc_main",
    sources = ["nbc_main.cc"],
    #headers = [],
    deplibs = [":nbc",
               "fastlib:fastlib"]
    )
