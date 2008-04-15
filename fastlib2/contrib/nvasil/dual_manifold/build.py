librule(name="dual_manifold",
    headers=["dual_manifold_engine.h", "dual_manifold_engine_impl.h", 
             "mvu_dot_prod_objective.h", "mvu_dot_prod_objective_impl.h"],
    deplibs=["fastlib:fastlib", "contrib/nvasil/l_bfgs:l_bfgs"])

binrule(name="dmanifold",
    sources=["main.cc"],
    deplibs=[":dual_manifold", "fastlib:fastlib"])

binrule(name="test",
    sources=["dual_manifold_engine_test.cc"],
    deplibs=[":dual_manifold", "fastlib:fastlib"])
