librule(
	name = "scf_solver",
	headers = ["scf_solver.h"],
	deplibs = ["fastlib:fastlib"],
	tests = ["scf_solver_test.cc"]
)

librule(
	name = "dual_tree_integrals",
	headers = ["dual_tree_integrals.h"],
	deplibs = ["fastlib:fastlib"],
	tests = ["dual_tree_integrals_test.cc"]
)

librule(
	name = "hf",
	headers = ["hf.h"],
	deplibs = [":scf_solver", ":dual_tree_integrals", "fastlib:fastlib"],
	tests = ["hf_test.cc"]
)

binrule(
	name = "dual_tree_integrals_main",
	headers = ["dual_tree_integrals.h", "naive_fock_matrix.h"],
	deplibs = ["fastlib:fastlib"],
	sources = ["dual_tree_integrals.cc"]
)