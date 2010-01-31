librule(
  	name="sparse",
		headers=["sparse_matrix.h", "conjugate_gradient.cc" ],
		deplibs=["fastlib:fastlib"]
		);

binrule(
		name="sparse_test",
		sources=["sparse_matrix_unit.cc"],
		headers=["sparse_matrix.h", "conjugate_gradient_impl.h"],
		deplibs=["fastlib:fastlib"]
		);


