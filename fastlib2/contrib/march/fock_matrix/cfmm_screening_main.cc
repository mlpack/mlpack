#include "fastlib/fastlib.h"
#include "eri.h"


const fx_entry_doc cfmm_screening_entries[] = {
{"centers", FX_REQUIRED, FX_STR, NULL, 
 "A file containing the centers of the basis functions.\n"},
{"exponents", FX_REQUIRED, FX_STR, NULL, 
"A file containing the exponents of the basis functions.\n"
"Must have the same number of rows as centers.\n"},
  {"threshold", FX_PARAM, FX_DOUBLE, NULL,
  "The threshold for cutting off a shell-pair.  Default: 10e-10\n"},
  {"centers_out", FX_PARAM, FX_STR, NULL,
  "The file to write the charge centers to.  Default: centers.csv \n"},
  {"exponents_out", FX_PARAM, FX_STR, NULL, 
  "The file to write the charge exponents to.  Default: exp.csv\n"}
};

const fx_module_doc cfmm_screening_main_doc = {
  cfmm_screening_entries, NULL, 
  "Selects significant charge distributions for use in the CFMM.\n"
};

int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, &cfmm_screening_main_doc);
  
  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);
  
  if (centers.n_cols() != exp_mat.n_cols()) {
    FATAL("Number of basis centers must equal number of exponents.\n");
  }
  
  double thresh;
  thresh = fx_param_double(root_mod, "threshold", 10e-10);
  
  index_t num_shells = centers.n_cols();
  ArrayList<BasisShell> shells;
  shells.Init(num_shells);
  
  for (index_t i = 0; i < num_shells; i++) {
  
    Vector new_cent;
    centers.MakeColumnVector(i, &new_cent);
  
    shells[i].Init(new_cent, exp_mat.ref(i,0), 0, i);
  
  } // for i
  
  
  ArrayList<ShellPair> shell_pairs;
  
  
  index_t num_shell_pairs = eri::ComputeShellPairs(&shell_pairs, shells, 
                                                   thresh);
                                                   
                
                                                   
  Matrix charge_centers;
  charge_centers.Init(3,num_shell_pairs);
  
  Matrix charge_exponents;
  charge_exponents.Init(num_shell_pairs, 1);
  
  for (index_t i = 0; i < num_shell_pairs; i++) {
  
    Vector cent_vec;
    double new_exp = eri::ComputeGPTCenter(shell_pairs[i].M_Shell().center(), 
                                           shell_pairs[i].M_Shell().exp(), 
                                           shell_pairs[i].N_Shell().center(), 
                                           shell_pairs[i].N_Shell().exp(), 
                                           &cent_vec);
  
    // add to output matrix
    charge_centers.CopyVectorToColumn(i, cent_vec);
    
    charge_exponents.set(i, 0, new_exp);
  
  } // for i
  
  const char* exp_out_file = fx_param_str(root_mod, "exponents_out", "exp.csv");
  data::Save(exp_out_file, charge_exponents);
  
  const char* centers_out_file = fx_param_str(root_mod, "centers_out", 
                                              "centers.csv");
  data::Save(centers_out_file, charge_centers);
  
  fx_result_int(root_mod, "num_charge_dists", num_shell_pairs);
  
  fx_done(root_mod);

} // main()
