librule(
    name="scf_solver",
    headers=["scf_solver.h"],
    sources=["scf_solver.cc"],
    deplibs=["fastlib:fastlib"],
)

binrule(
    name="scf_main",
    sources=["scf_main.cc"],
    deplibs=["fastlib:fastlib", "contrib/march/fock_matrix/fock_impl:oeints", "contrib/march/fock_matrix/cfmm:cfmm_coulomb", "contrib/march/fock_matrix/naive:naive_fock_matrix", "contrib/march/fock_matrix/link:link_exchange", "contrib/march/fock_matrix/prescreening:schwartz_prescreening", "contrib/march/fock_matrix/multi_tree:multi_tree_fock"],
)
