#include "fastlib/fastlib.h"
#include "nmf_engine.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module *nmf_module=fx_submodule(fx_root, "/engine");
	NmfEngine engine;
	fx_set_param_str(nmf_module, "data_file", "v.csv");
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 0.1);
	engine.Init(nmf_module);
  engine.ComputeNmf();
	Matrix w_mat;
	Matrix h_mat;
  engine.GetW(&w_mat);
	engine.GetH(&h_mat);
  data::Save("W.csv", w_mat);
	data::Save("H.csv", h_mat);
}
