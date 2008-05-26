/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/23/2008 11:25:40 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "fastlib/fastlib.h"
#include "sdp_nmf_engine.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module *nmf_module=fx_submodule(fx_root, "/engine");
  SdpNmfEngine<SmallSdpNmf> engine;
  fx_set_param_str(nmf_module, "data_file", "v.csv");
  //    "/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv");
  fx_set_param_int(nmf_module, "new_dim", 2);
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 1);
  fx_set_param_double(nmf_module, "l_bfgs/gamma", 2);
  fx_set_param_double(nmf_module, "l_bfgs/wolfe_sigma2", 0.9999);
  fx_set_param_double(nmf_module, "l_bfgs/use_default_termination", false);
  fx_set_param_int(nmf_module, "l_bfgs/mem_bfgs", 10); 
  fx_set_param_double(nmf_module, "optfun/desired_duality_gap", 1e-5); 
	fx_set_param_double(nmf_module, "optfun/norm_grad_tolerance", 1e-10);
  engine.Init(nmf_module);
  engine.ComputeNmf();
	Matrix w_mat;
	Matrix h_mat;
  engine.GetW(&w_mat);
	engine.GetH(&h_mat);
  data::Save("W.csv", w_mat);
	data::Save("H.csv", h_mat);
}
