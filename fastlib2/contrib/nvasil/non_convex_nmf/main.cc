#include "fastlib/fastlib.h"
#include "nmf_engine.h"

 //typedef NmfEngine<ClassicNmfObjective> Engine;
 //typedef NmfEngine<BigSdpNmfObjectiveMaxVarIsometric> Engine;
 typedef NmfEngine<BigSdpNmfObjectiveMaxFurthestIsometric> Engine;
 //typedef NmfEngine<NmfObjectiveIsometric> Engine;


int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module *nmf_module=fx_submodule(fx_root, "/engine");
  Engine *engine;
  fx_set_param_str(nmf_module, "data_file", 
   // "/net/hg200/nvasil/dataset/corel/ColorHistogram_1000_30.csv");
   // "/net/hg200/nvasil/dataset/amlall/amlall.csv");
   // "/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_472.csv");
    "/net/hg200/nvasil/dataset/teapots/teapots_small.csv");
	fx_set_param_int(nmf_module, "sdp_rank", 3);
  fx_set_param_int(nmf_module, "new_dimension", 20);
  fx_set_param_int(nmf_module, "optfun/knns", 7);
  fx_set_param_double(nmf_module, "optfun/grad_tolerance", 1e-6);
  fx_set_param_double(nmf_module, "optfun/feasibility_error",40);
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 10);
	fx_set_param_double(nmf_module, "l_bfgs/gamma", 5);
 	fx_set_param_double(nmf_module, "l_bfgs/step_size", 3);
  fx_set_param_int(nmf_module, "l_bfgs/mem_bfgs", 5);
	fx_set_param_bool(nmf_module, "l_bfgs/silent", false);
  fx_set_param_bool(nmf_module, "l_bfgs/use_default_termination", false);
  for(index_t i=0; i<1; i++) {
    engine = new Engine();
    engine->Init(nmf_module);
    engine->ComputeNmf();
	  Matrix w_mat;
	  Matrix h_mat;
    engine->GetW(&w_mat);
	  engine->GetH(&h_mat);
    data::Save("W.csv", w_mat);
	  data::Save("H.csv", h_mat);
    delete engine;
  }
  fx_done(fx_root);
}
