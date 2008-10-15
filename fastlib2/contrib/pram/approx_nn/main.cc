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
    = fx_submodule(root, "ann");

  fx_param_int(ann_module, "epsilon", 5);
  fx_param_double(ann_module, "alpha", 0.90);

  ArrayList<index_t> nac, exc, apc;
  ArrayList<double> din, die, dia;

  // Naive computation
  fx_timer_start(ann_module, "naive_init");
  naive_nn.InitNaive(qdata, rdata, 1);
  fx_timer_stop(ann_module, "naive_init");

  fx_timer_start(ann_module, "naive");
  naive_nn.ComputeNaive(&nac, &din);
  fx_timer_start(ann_module, "naive");

  // Exact computation
  fx_timer_start(ann_module, "exact_init");
  exact_nn.Init(qdata, rdata, ann_module);
  fx_timer_stop(ann_module, "exact_init");

  fx_timer_start(ann_module, "exact");
  exact_nn.ComputeNeighbors(&exc, &die);
  fx_timer_stop(ann_module, "exact");

  // Approximate computation
  fx_timer_start(ann_module, "approx_init");
  approx_nn.Init(qdata, rdata, ann_module);
  fx_timer_stop(ann_module, "approx_init");

  ArrayList<double> prob;
  fx_timer_start(ann_module, "approx");
  approx_nn.ComputeApprox(&apc, &dia, &prob);
  fx_timer_stop(ann_module, "approx");

  fx_done(fx_root);
}
