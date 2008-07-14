#include "fastlib/fastlib.h"
#include "nmf_engine.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module *nmf_module=fx_submodule(fx_root, "/engine");
  NmfEngine<ClassicNmfObjective> engine;
  //NmfEngine<BigSdpNmfObjectiveMinVarDiagonalDominance> engine;
  // NmfEngine<BigSdpNmfObjectiveMaxVarIsometric> engine;
  fx_set_param_str(nmf_module, "data_file", 
     "/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv");
  //"/net/hg200/nvasil/dataset/amlall/amlall.csv");
	fx_set_param_int(nmf_module, "sdp_rank", 3);
  fx_set_param_int(nmf_module, "new_dim", 30);
  fx_set_param_int(nmf_module, "knns", 3);
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 10);
	fx_set_param_double(nmf_module, "l_bfgs/gamma", 3);
	fx_set_param_double(nmf_module, "l_bfgs/norm_grad_tolerance", 1e-7);
	fx_set_param_double(nmf_module, "l_bfgs/desired_feasibility", 0.001);
  fx_set_param_double(nmf_module, "l_bfgs/feasibility_tolerance", 1e-7);
  fx_set_param_int(nmf_module, "l_bfgs/mem_bfgs", 8);
	fx_set_param_bool(nmf_module, "l_bfgs/silent", false);
  engine.Init(nmf_module);
  for(index_t i=0; i<1; i++) {
    engine.ComputeNmf();
	  Matrix w_mat;
	  Matrix h_mat;
    engine.GetW(&w_mat);
	  engine.GetH(&h_mat);
    data::Save("W.csv", w_mat);
	  data::Save("H.csv", h_mat);
  }
  fx_done(fx_root);
}
