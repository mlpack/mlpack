#include <string>
#include "approx_nn.h"

const fx_entry_doc approx_nn_main_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_REQUIRED, FX_STR, NULL,
   " A file containing the query set"
   " (defaults to the reference set).\n"},
  {"donaive", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the naive computation(defaults to false).\n"},
  {"doexact", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the exact computation"
   "(defaults to true).\n"},
  {"doapprox", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the approximate computation(defaults to true).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc approx_nn_main_submodules[] = {
  {"ann", &approx_nn_doc,
   " Responsible for doing approximate nearest neighbor"
   " search using sampling on kd-trees.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc approx_nn_main_doc = {
  approx_nn_main_entries, approx_nn_main_submodules,
  "This is a program to test run the approx "
  " nearest neighbors using sampling on kd-trees.\n"
  "It performs the exact, approximate"
  " and the naive computation.\n"
};


int main (int argc, char *argv[]) {
  fx_module *root
    = fx_init(argc, argv, &approx_nn_main_doc);

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
  if (fx_param_bool(root, "donaive", false)) {
    ApproxNN naive_nn;
    NOTIFY("Naive");
    NOTIFY("Init");
    fx_timer_start(ann_module, "naive_init");
    naive_nn.InitNaive(qdata, rdata, 1);
    fx_timer_stop(ann_module, "naive_init");

  //   NOTIFY("Compute");
  //   fx_timer_start(ann_module, "naive");
  //   naive_nn.ComputeNaive(&nac, &din);
  //   fx_timer_start(ann_module, "naive");
  }

  // Exact computation
  if (fx_param_bool(root, "doexact", true)) {
    ApproxNN exact_nn;
    NOTIFY("Exact");
    NOTIFY("Init");
    fx_timer_start(ann_module, "exact_init");
    exact_nn.Init(qdata, rdata, ann_module);
    fx_timer_stop(ann_module, "exact_init");

  //   NOTIFY("Compute");
  //   fx_timer_start(ann_module, "exact");
  //   exact_nn.ComputeNeighbors(&exc, &die);
  //   fx_timer_stop(ann_module, "exact");
  }

  // Approximate computation
  if (fx_param_bool(root, "doapprox", true)) {
    ApproxNN approx_nn;
    NOTIFY("Approx");
    NOTIFY("Init");
    fx_timer_start(ann_module, "approx_init");
    approx_nn.InitApprox(qdata, rdata, ann_module);
    fx_timer_stop(ann_module, "approx_init");

  //   NOTIFY("Compute");
  //   ArrayList<double> prob;
  //   fx_timer_start(ann_module, "approx");
  //   approx_nn.ComputeApprox(&apc, &dia, &prob);
  //   fx_timer_stop(ann_module, "approx");
  }

  fx_param_bool(root, "fx/silent", true);
  fx_done(fx_root);
}
