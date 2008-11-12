librule(name="opt++",
        headers=["optimizer.h"],
        tests=["optim_test.cc"],
        deplibs=["fastlib:fastlib", "opt++/lib/libnewmat.a",
        "opt++/lib/libopt.a"]
    )
