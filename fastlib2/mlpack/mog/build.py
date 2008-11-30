
librule(
    name = "mog_em",
    sources = ["mog_em.cc"],
    headers = ["mog_em.h",
               "phi.h",
               "math_functions.h"],
    deplibs = ["fastlib:fastlib"],
    )

librule(
    name = "mog_l2e",
    sources = ["mog_l2e.cc"],
    headers = ["mog_l2e.h",
               "phi.h"],
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "mog_em_main",
    sources = ["mog_em_main.cc"],
    deplibs = [":mog_em",
               "fastlib:fastlib"]
    )

binrule(
    name = "mog_l2e_main",
    sources = ["mog_l2e_main.cc"],
    deplibs = [":mog_l2e",
               "mlpack/optimization:opt++",
               "fastlib:fastlib"]
    )
