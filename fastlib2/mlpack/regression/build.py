librule(name="ridge_regreesion",
        headers=["ridge_regression.h", "ridge_regression_impl.h"],
        tests=["ridge_regression_test.cc"],
        deplibs=["fastlib:fastlib"]
       )
