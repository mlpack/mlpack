#include <string>
#include "approx_nn.h"

const fx_entry_doc approx_nn_main_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_REQUIRED, FX_STR, NULL,
   " A file containing the query set"
   " (defaults to the reference set).\n"},
  {"Init", FX_TIMER, FX_CUSTOM, NULL,
   " Nik's tree code init.\n"},
  {"Compute", FX_TIMER, FX_CUSTOM, NULL,
   " Nik's tree code compute.\n"},
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

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compare_neighbors(ArrayList<index_t>*, ArrayList<double>*, 
                       ArrayList<index_t>*, ArrayList<double>*);

void count_mismatched_neighbors(ArrayList<index_t>*, ArrayList<double>*, 
				  ArrayList<index_t>*, ArrayList<double>*);

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

//   AllkNN allknn;
//   ArrayList<index_t> neighbor_indices;
//   ArrayList<double> dist_sq;
//   fx_timer_start(root, "Init");
//   allknn.Init(qdata, rdata, 20, 4);
//   fx_timer_stop(root, "Init");
//   fx_timer_start(root, "Compute");
//   allknn.ComputeNeighbors(&neighbor_indices, &dist_sq);
//   fx_timer_stop(root, "Compute");

  struct datanode *ann_module
    = fx_submodule(root, "ann");

  ArrayList<index_t> nac, exc, apc;
  ArrayList<double> din, die, dia;

  // Naive computation
  if (fx_param_bool(root, "donaive", false)) {
    ApproxNN naive_nn;
    NOTIFY("Brute computation");
    NOTIFY("Initializing....");
    fx_timer_start(ann_module, "naive_init");
    naive_nn.InitNaive(qdata, rdata, 1);
    fx_timer_stop(ann_module, "naive_init");
    NOTIFY("Initialized.");

    NOTIFY("Computing Neighbors.....");
    fx_timer_start(ann_module, "naive");
    naive_nn.ComputeNaive(&nac, &din);
    fx_timer_stop(ann_module, "naive");
    NOTIFY("Neighbors Computed.");
  }

  // Exact computation
  if (fx_param_bool(root, "doexact", true)) {
    ApproxNN exact_nn;
    NOTIFY("Exact using Single Tree");
    NOTIFY("Initializing....");
    fx_timer_start(ann_module, "exact_init");
    exact_nn.Init(qdata, rdata, ann_module);
    fx_timer_stop(ann_module, "exact_init");
    NOTIFY("Initialized.");

    NOTIFY("Computing Neighbors.....");
    fx_timer_start(ann_module, "exact");
    exact_nn.ComputeNeighbors(&exc, &die);
    fx_timer_stop(ann_module, "exact");
    NOTIFY("Neighbors Computed.");
  }

//   compare_neighbors(&neighbor_indices, &dist_sq, &exc, &die);

  // Approximate computation
  if (fx_param_bool(root, "doapprox", true)) {
    ApproxNN approx_nn;
    NOTIFY("Rank Approximate using Single Tree");
    NOTIFY("Initializing....");
    fx_timer_start(ann_module, "approx_init");
    approx_nn.InitApprox(qdata, rdata, ann_module);
    fx_timer_stop(ann_module, "approx_init");
    NOTIFY("Initialized.");

    NOTIFY("Computing Neighbors.....");
    fx_timer_start(ann_module, "approx");
    approx_nn.ComputeApprox(&apc, &dia);
    fx_timer_stop(ann_module, "approx");
    NOTIFY("Neighbors Computed.");
  }
  
  //  count_mismatched_neighbors(&exc, &die, &apc, &dia);

  fx_done(fx_root);
}

void compare_neighbors(ArrayList<index_t> *a, 
                       ArrayList<double> *da,
                       ArrayList<index_t> *b, 
                       ArrayList<double> *db) {
  
  NOTIFY("Comparing results for %"LI"d queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();

  for(index_t i = 0; x != y; x++, z++, i++) {
    DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
                      "point %"LI"d brute: %"LI"d:%lf fast: %"LI"d:%lf",
                      i, *z, (*db)[i], *x, (*da)[i]);
  }
}

void count_mismatched_neighbors(ArrayList<index_t> *a, 
				ArrayList<double> *da,
				ArrayList<index_t> *b, 
				ArrayList<double> *db) {

  NOTIFY("Comparing results for %"LI"d queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();
  index_t count_mismatched = 0;

  for(index_t i = 0; x != y; x++, z++, i++) {
    if (*x != *z || (*da)[i] != (*db)[i]) {
      ++count_mismatched;
    }
  }
  NOTIFY("%"LI"d/%"LI"d errors", count_mismatched, a->size());
}
