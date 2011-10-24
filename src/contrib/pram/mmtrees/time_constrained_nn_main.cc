#include <string>
#include "time_constrained_nn.h"

const fx_entry_doc time_constrained_nn_main_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   " A file containing the test query set.\n"},
  {"rank_file", FX_REQUIRED, FX_STR, NULL,
   "The file containing the ranks of the reference points "
   "for to the queries.\n"},
  {"rank_file_too_big", FX_PARAM, FX_BOOL, NULL,
   "This boolean parameter lets the program know that "
   "the rank file is too big to load in RAM and so "
   "do something interesting (defaults to false).\n"},
  {"result", FX_PARAM, FX_STR, NULL,
   " A file where some results would be written into."
   " (defaults to 'results.txt').\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc time_constrained_nn_main_submodules[] = {
  {"tc_nn", &time_constrained_nn_doc,
   " Responsible for doing time constrained nearest neighbor"
   " search using kd-trees.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc time_constrained_nn_main_doc = {
  time_constrained_nn_main_entries, time_constrained_nn_main_submodules,
  "This is a program to test run the time constrained "
  "nearest neighbor search outputing the rank error "
  "with increasing number of leaves visited."
};

int main (int argc, char *argv[]) {
  fx_module *root
    = fx_init(argc, argv, &time_constrained_nn_main_doc);

  Matrix qdata, rdata;
  std::string rfile = fx_param_str_req(root, "r");
  NOTIFY("Loading files...");
  data::Load(rfile.c_str(), &rdata);

  if (fx_param_exists(root, "q")) {
    std::string qfile = fx_param_str_req(root, "q");
    data::Load(qfile.c_str(), &qdata);
  } else {
    qdata.Copy(rdata);
  }

  NOTIFY("Loading rank file, might take time...");
  Matrix rank_mat;
  std::string rank_file = fx_param_str_req(root, "rank_file");
  if (!fx_param_bool(root, "rank_file_too_big", false)) {
    data::Load(rank_file.c_str(), &rank_mat);
  } else {
    rank_mat.Init(0,0);
  }

  NOTIFY("Files loaded...");
  NOTIFY("R(%zud, %zud), Q(%zud, %zud)",
	 rdata.n_rows(), rdata.n_cols(), 
	 qdata.n_rows(), qdata.n_cols());
  if (!fx_param_bool(root, "rank_file_too_big", false)) {
    NOTIFY("Rank Matrix: %zud x %zud",
	   rank_mat.n_rows(), rank_mat.n_cols());
  }
    // NOTIFY("Test Q(%zud, %zud)",
    //  test_qdata.n_rows(), test_qdata.n_cols());

  // exit(1);

  struct datanode *nn_module
    = fx_submodule(root, "tc_nn");

  ArrayList<size_t> max_error, min_error;
  ArrayList<double> mean_error, std_error;

  // Exact computation
  // if (fx_param_bool(root, "doexact", true)) {
  TCNN tc_nn;
  NOTIFY("Time constrained analysis using Single Tree");

  // building the kd-tree for the reference trees
  NOTIFY("Initializing....");
  fx_timer_start(nn_module, "init");
  tc_nn.Init(rdata, nn_module);
  fx_timer_stop(nn_module, "init");
  NOTIFY("Initialized.");


  // This function computes the exact neighbors all 
  // the way to the end and computes the rank error
  // at every leaf visited to show how the rank error
  // falls with the number of the leafs visited.
  NOTIFY("Computing the neighbors exactly....");
  fx_timer_start(nn_module, "exact");
  if (!fx_param_bool(root, "rank_file_too_big", false)) {
    tc_nn.InitQueries(qdata, rank_mat);
  } else {
    tc_nn.InitQueries(qdata, rank_file);
  }
  tc_nn.ComputeNeighborsSequential(&mean_error, &std_error,
				   &max_error, &min_error);
  fx_timer_stop(nn_module, "exact");

  DEBUG_ASSERT(mean_error.size() == std_error.size());
  DEBUG_ASSERT(max_error.size() == min_error.size());
  DEBUG_ASSERT(mean_error.size() == min_error.size());


  // printing the results out into a file.
  if (fx_param_exists(root, "result")) {
    std::string result_file = fx_param_str_req(root, "result");
    FILE *fp = fopen(result_file.c_str(), "w");

    for (size_t i = 0; i < mean_error.size(); i++) 
      fprintf(fp, "%lg,%lg,%zud,%zud\n", mean_error[i],
	      std_error[i], min_error[i], max_error[i]);

    fclose(fp);
  }

  fx_done(fx_root);
  return 1;
}
