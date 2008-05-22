#include "fastlib/fastlib.h"
#include "nmf_engine.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module *nmf_module=fx_submodule(fx_root, "/engine");
	NmfEngine engine;
	fx_set_param_str(nmf_module, "data_file", 
      "/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv");
	fx_set_param_int(nmf_module, "sdp_rank",1 );
  fx_set_param_int(nmf_module, "new_dim", 20);
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 0.010);
	fx_set_param_double(nmf_module, "l_bfgs/norm_grad_tolerance", 10);
	fx_set_param_double(nmf_module, "l_bfgs/desired_feasibility", 1);
  fx_set_param_double(nmf_module, "l_bfgs/feasibility_tolerance", 0.01);
  fx_set_param_int(nmf_module, "l_bfgs/mem_bfgs", 5);
  engine.Init(nmf_module);
  engine.ComputeNmf();
	Matrix w_mat;
	Matrix h_mat;
  engine.GetW(&w_mat);
	engine.GetH(&h_mat);
  data::Save("W.csv", w_mat);
	data::Save("H.csv", h_mat);
}
