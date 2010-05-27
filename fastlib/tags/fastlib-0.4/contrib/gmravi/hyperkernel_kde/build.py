librule(
    name="hk_ippc_lib",
    sources=[],
    headers=["hyperkernels.h","interior_point_pred_corr2.h","ichol_dynamic.h","dte.h","special_la.h"],
    deplibs=["fastlib:fastlib_int","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )


binrule(
    name="hk_ippc",
    sources=["interior_point_pred_corr_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":hk_ippc_lib","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )

binrule(
    name="varying_effects",
    sources=["varying_effects.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":hk_ippc_lib","mlpack/kde:dualtree_kde"]
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

librule(
    name="adaptive_kde_lib",
    sources=[],
    headers=["adaptive_kde.h"],
    deplibs=["fastlib:fastlib_int","mlpack/fastica:fastica_lib"]
    )


binrule(
    name="adaptive_kde",
    sources=["adaptive_kde_main.cc"],
    headers=[""],
    deplibs=["fastlib:fastlib_int",":adaptive_kde_lib","mlpack/fastica:fastica_lib"]
    )

librule(
    name="variable_kde_lib",
    sources=[],
    headers=["variable_nn_based_kde.h"],
    deplibs=["fastlib:fastlib_int","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )


binrule(
    name="variable_nn_based_kde",
    sources=["variable_nn_based_kde_main.cc"],
     headers=["variable_nn_based_kde.h"],
    deplibs=["fastlib:fastlib_int",":variable_kde_lib","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )


librule(
    name="nn_kde_lib",
    sources=[],
    headers=["nn_kde.h"],
    deplibs=["fastlib:fastlib_int","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )


binrule(
    name="nn_kde",
    sources=["nn_kde_main.cc"],
     headers=["nn_kde.h"],
    deplibs=["fastlib:fastlib_int",":nn_kde_lib","mlpack/kde:dualtree_kde","mlpack/fastica:fastica_lib"]
    )
