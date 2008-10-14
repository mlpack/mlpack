#include <string>
#include "approx_nn.h"

int main (int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, NULL);

  ApproxNN approx_nn, naive_nn, exact_nn;
  Matrix qdata, rdata;
  std::string qfile = fx_param_str_req(root, "q");
  std::string rfile = fx_param_str_req(root, "r");
  NOTIFY("Loading files...");
  data::Load(qfile.c_str(), &qdata);
  data::Load(rfile.c_str(), &rdata);
  NOTIFY("File loaded...");

  struct datanode *ann_module
    fx_submodule(root, "ann");

  fx_param_int(ann_module, "epsilon", 5);
  fx_param_double(ann_module, "alpha", 0.90);

  fx_timer_start(ann_module, "naive_init");
  naive_nn.InitNaive(qdata, rdata, 1);
  fx_timer_stop(ann_module, "naive_init");

  exact_nn.Init(qdata, rdata, ann_module);

  approx_nn.Init(qdata, rdata, ann_module);
  NOTIFY("Computing Neighbors...");
  allnn.ComputeNeighbors(NULL, NULL);
  NOTIFY("Neighbors Computed...");
  fx_done(fx_root);
}
