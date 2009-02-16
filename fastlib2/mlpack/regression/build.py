librule(name="ridge_regression",
        headers=["ridge_regression.h", "ridge_regression_impl.h"],
        tests=["ridge_regression_test.cc"],
        deplibs=["fastlib:fastlib"]
       )
binrule(name="ridge",
        sources=["ridge_main.cc"],
        deplibs=[":ridge_regression"]
    )
