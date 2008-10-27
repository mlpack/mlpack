#librule(
 #   name="hk_convex_ise_lib",
  #  sources=[],
   # headers=["hyperkernels.h","hk_convex_ise.h","ichol.h","dte.h",
    #         "special_linear_algebra.h"],
   # deplibs=["fastlib:fastlib_int"]
   # )


#binrule(
 #   name="hk_convex_ise",
  #  sources=["hk_convex_ise_main.cc"],
   # headers=[""],
    #deplibs=["fastlib:fastlib_int",":hk_convex_ise_lib"]
   # )



librule(
    name="hk_ippc_lib",
    sources=[],
    headers=["hyperkernels.h","interior_point_pred_corr.h","ichol_dynamic.h","dte.h",
             "special_la.h"],
    deplibs=["fastlib:fastlib_int","mlpack/kde:kde_cv_bin"]
    )


binrule(
    name="hk_ippc",
    sources=["interior_point_pred_corr_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":hk_ippc_lib","mlpack/kde:kde_cv_bin"]
    )

librule(
    name="test_lib",
    sources=[],
    headers=["special_linear_algebra.h"],
    deplibs=["fastlib:fastlib_int"]
    )


binrule(
    name="test",
    sources=["special_la_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":hk_ippc_lib"]
    )

librule(
    name="test_ichol_lib",
    sources=[],
    headers=["ichol_dynamic.h"],
    deplibs=["fastlib:fastlib_int"]
    )


binrule(
    name="test_ichol",
    sources=["test_ichol_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":test_ichol_lib"]
    )
