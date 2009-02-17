#include "link.h"

const fx_entry_doc link_main_entries[] = {
{"shell_pair_cutoff", FX_PARAM, FX_DOUBLE, NULL, 
  "The threshold for a shell pair to be counted as \"significant\".\n"
  "Default: equal to the ERI threshold.\n"},
  FX_ENTRY_DOC_DONE
}; 


const fx_module_doc link_main_doc = {
  link_main_entries, NULL, "Main for running LinK exchange method.\n"
};




int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, &link_main_doc);

  fx_module* link_mod = fx_submodule(root_mod, "link");

  double thresh = fx_param_double(root_mod, "threshold", 10e-10);

  const char* centers_file = fx_param_str_req(root_mod, "centers");
  Matrix centers_mat;
  data::Load(centers_file, &centers_mat);
  
  Vector exp_vec;
  if (fx_param_exists(root_mod, "exponents")) {
    const char* exp_file = fx_param_str_req(root_mod, "exponents");
    Matrix exp_mat;
    data::Load(exp_file, &exp_mat);
    exp_mat.MakeColumnVector(0, &exp_vec);
  }
  else {
    exp_vec.Init(centers_mat.n_cols());
    exp_vec.SetAll(1.0);
  }

  Vector mom_vec;
  if (fx_param_exists(root_mod, "momenta")) {
    Matrix mom_mat;
    const char* mom_file = fx_param_str_req(root_mod, "momenta");
    data::Load(mom_file, &mom_mat);
    mom_mat.MakeColumnVector(0, &mom_vec);
  }
  else {
    mom_vec.Init(centers_mat.n_cols());
    mom_vec.SetAll(0.0);
  }
  
  Matrix density_mat;
  if (fx_param_exists(root_mod, "density")) {
    const char* density_file = fx_param_str_req(root_mod, "density");
    data::Load(density_file, &density_mat);
  }
  else {
    density_mat.Init(centers_mat.n_cols(), centers_mat.n_cols());
    density_mat.SetAll(1.0);
  }

  Link link_algorithm;
  link_algorithm.Init(link_mod, thresh, centers_mat, exp_vec, mom_vec, 
                      density_mat);
                      
  fx_timer_start(root_mod, "link_time");
  link_algorithm.ComputeFockMatrix();
  fx_timer_stop(root_mod, "link_time");

  fx_done(root_mod);

  return 0;

} // main()