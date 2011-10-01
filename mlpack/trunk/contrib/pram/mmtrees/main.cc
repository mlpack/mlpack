#include <string>
#include "approx_nn.h"

const fx_entry_doc approx_nn_main_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"tq", FX_PARAM, FX_STR, NULL,
   " A file containing the training query set"
   " (defaults to part of the reference set).\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   " A file containing the test query set.\n"},
  {"perc_ref", FX_PARAM, FX_INT, NULL,
   " The percentage of the reference set to be "
   "used as a training query set (integers between 1-100)."
   " Defaults to 100."},
  {"Init", FX_TIMER, FX_CUSTOM, NULL,
   " Nik's tree code init.\n"},
  {"Compute", FX_TIMER, FX_CUSTOM, NULL,
   " Nik's tree code compute.\n"},
  {"doexact", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the exact computation"
   "(defaults to true).\n"},
  {"doapprox", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the approximate computation(defaults to true).\n"},
  {"result", FX_PARAM, FX_STR, NULL,
   " A file where some results would be written into."
   " (defaults to 'results.txt').\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc approx_nn_main_submodules[] = {
  {"ann", &approx_nn_doc,
   " Responsible for doing approximate nearest neighbor"
   " search using kd-trees.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc approx_nn_main_doc = {
  approx_nn_main_entries, approx_nn_main_submodules,
  "This is a program to test run the approx "
  " nearest neighbors using kd-trees.\n"
  "It performs the exact and approximate"
  " computation by just traveling down the tree "
  "and returning the neighbor candidate as the "
  "approximate answer."
};

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compare_neighbors(ArrayList<size_t>*, ArrayList<double>*, 
                       ArrayList<size_t>*, ArrayList<double>*);

void count_mismatched_neighbors(ArrayList<size_t>*, ArrayList<double>*, 
				  ArrayList<size_t>*, ArrayList<double>*);

void make_train_subset_from_ref(Matrix &ref, size_t perc_ref,
				Matrix *train_q);

int main (int argc, char *argv[]) {
  fx_module *root
    = fx_init(argc, argv, &approx_nn_main_doc);

  Matrix test_qdata, train_qdata, rdata;
  std::string rfile = fx_param_str_req(root, "r");
  NOTIFY("Loading files...");
  data::Load(rfile.c_str(), &rdata);

  if (fx_param_exists(root, "tq")) {
    std::string qfile = fx_param_str_req(root, "tq");
    data::Load(qfile.c_str(), &train_qdata);
  } else {
    size_t perc_ref = fx_param_int(root, "perc_ref", 100);
    if (perc_ref == 100) 
      train_qdata.Copy(rdata);
    else
      make_train_subset_from_ref(rdata, perc_ref, &train_qdata);
  }

  if (fx_param_exists(root, "q")) {
    std::string qfile = fx_param_str_req(root, "q");
    data::Load(qfile.c_str(), &test_qdata);
  } else {
    test_qdata.Copy(rdata);
  }

  NOTIFY("File loaded...");
  NOTIFY("R(%zud, %zud), Train Q(%zud, %zud)",
	 rdata.n_rows(), rdata.n_cols(), 
	 train_qdata.n_rows(), train_qdata.n_cols());
  NOTIFY("Test Q(%zud, %zud)", test_qdata.n_rows(), test_qdata.n_cols());

  // exit(0);
//   AllkNN allknn;
//   ArrayList<size_t> neighbor_indices;
//   ArrayList<double> dist_sq;
//   fx_timer_start(root, "Init");
//   allknn.Init(qdata, rdata, 20, 4);
//   fx_timer_stop(root, "Init");
//   fx_timer_start(root, "Compute");
//   allknn.ComputeNeighbors(&neighbor_indices, &dist_sq);
//   fx_timer_stop(root, "Compute");

  struct datanode *nn_module
    = fx_submodule(root, "ann");

  ArrayList<size_t> exc, apc;
  ArrayList<double> die, dia;

  // Exact computation
  if (fx_param_bool(root, "doexact", true)) {
    ApproxNN exact_nn;
    NOTIFY("Exact using Single Tree");

    NOTIFY("Training Neighbors.....");
    NOTIFY("Initializing....");
    fx_timer_start(nn_module, "exact_init");
    exact_nn.InitTrain(train_qdata, rdata, nn_module);
    fx_timer_stop(nn_module, "exact_init");
    NOTIFY("Initialized.");

    fx_timer_start(nn_module, "exact");
    exact_nn.TrainNeighbors(); //, &apc, &dia);
    fx_timer_stop(nn_module, "exact");
    NOTIFY("Training Computed.");

    exit(0);

    NOTIFY("Testing Neighbors.....");
    NOTIFY("Initializing....");
    fx_timer_start(nn_module, "exact_init");
    exact_nn.InitTest(test_qdata);
    fx_timer_stop(nn_module, "exact_init");
    NOTIFY("Initialized.");

    fx_timer_start(nn_module, "exact");
    exact_nn.TestNeighbors(&exc, &die, &apc, &dia);
    fx_timer_stop(nn_module, "exact");
    NOTIFY("Testing Computed.");

//     std::string result_file = fx_param_str(root, "result", "result.txt");
//     FILE *fp = fopen(result_file.c_str(), "w");
//     for (size_t i = 0; i < exc.size(); i++) {
//       fprintf(fp,"%zud,%lg,", //%zud,%lg\n",
// 	      exc[i], die[i]); //, apc[i], dia[i]);
//     }
//     fclose(fp);
  }

//   compare_neighbors(&neighbor_indices, &dist_sq, &exc, &die);

  // Approximate computation
//   if (fx_param_bool(root, "doapprox", false)) {
//     ApproxNN approx_nn;
//     NOTIFY("Rank Approximate using Single Tree");
//     NOTIFY("Initializing....");
//     fx_timer_start(ann_module, "approx_init");
//     approx_nn.InitApprox(qdata, rdata, ann_module);
//     fx_timer_stop(ann_module, "approx_init");
//     NOTIFY("Initialized.");

//     NOTIFY("Computing Neighbors.....");
//     fx_timer_start(ann_module, "approx");
//     approx_nn.ComputeApprox(&apc, &dia);
//     fx_timer_stop(ann_module, "approx");
//     NOTIFY("Neighbors Computed.");
//   }
  
  //  count_mismatched_neighbors(&exc, &die, &apc, &dia);

  fx_done(fx_root);
}

void make_train_subset_from_ref(Matrix &ref, size_t perc_ref,
				Matrix *train_q) {
  size_t num_points = ref.n_cols();
  size_t num_train = (num_points * perc_ref) / 100;
  size_t step = num_points / num_train;

  train_q->Init(ref.n_rows(), num_train);
  size_t j = 0;

  for (size_t i = 0; i < num_points; i = i+step, j++) {
    Vector a, b;
    ref.MakeColumnVector(i, &a);
    train_q->MakeColumnVector(j, &b);

    b.CopyValues(a);
  }

  DEBUG_ASSERT(j == num_train);
  // train_q->PrintDebug("Train Q");

  //exit(1);
  return;

}



void compare_neighbors(ArrayList<size_t> *a, 
                       ArrayList<double> *da,
                       ArrayList<size_t> *b, 
                       ArrayList<double> *db) {
  
  NOTIFY("Comparing results for %zud queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  size_t *x = a->begin();
  size_t *y = a->end();
  size_t *z = b->begin();

  for(size_t i = 0; x != y; x++, z++, i++) {
    DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
                      "point %zud brute: %zud:%lf fast: %zud:%lf",
                      i, *z, (*db)[i], *x, (*da)[i]);
  }
}

void count_mismatched_neighbors(ArrayList<size_t> *a, 
				ArrayList<double> *da,
				ArrayList<size_t> *b, 
				ArrayList<double> *db) {

  NOTIFY("Comparing results for %zud queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  size_t *x = a->begin();
  size_t *y = a->end();
  size_t *z = b->begin();
  size_t count_mismatched = 0;

  for(size_t i = 0; x != y; x++, z++, i++) {
    if (*x != *z || (*da)[i] != (*db)[i]) {
      ++count_mismatched;
    }
  }
  NOTIFY("%zud/%zud errors", count_mismatched, a->size());
}
