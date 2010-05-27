librule(name = "ridge_regression",
        headers = ["ridge_regression.h",
                   "ridge_regression_impl.h",
                   "ridge_regression_util.h"],
        tests = ["ridge_regression_test.cc"],
        deplibs = ["fastlib:fastlib", "mlpack/quicsvd:quicsvd"]
       )

binrule(name = "ridge",
        sources = ["ridge_main.cc"],
        deplibs = [":ridge_regression"]
    )

binrule(name = "dataset_preprocess",
        sources = ["dataset_preprocess.cc"],
        deplibs = ["fastlib:fastlib"]
        )
