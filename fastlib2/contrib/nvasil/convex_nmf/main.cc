#include "fastlib/fastlib.h"
#include "nmf_engine.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
	fx_module nmf_module=fx_submodule(fx_root, "/engine");
	NmfEngine engine;
	engine.Init(nmf_mofule);
  engine.ComputeNmf();
	Matrix w_mat;
	Matrix h_mat;
  engine.GetW(&w_mat);
	engine.GetH(&h_mat);
  data::Save("W.csv", w_mat);
	data::Save("H.csv", h_mat);
}
