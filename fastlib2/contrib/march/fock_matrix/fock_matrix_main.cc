#include "fastlib/fastlib.h"



const fx_entry_doc fock_matrix_main_entries[] = {
{"centers", FX_REQUIRED, FX_STR, NULL, 
  "A file containing the centers of the basis functions.\n"},
{"exponents", FX_REQUIRED, FX_STR, NULL, 
  "A file containing the exponents of the basis functions.\n"
  "Must have the same number of rows as centers.\n"},
{"density", FX_PARAM, FX_STR, NULL, 
  "A file containing the density matrix.  If it is not provided, an all-ones\n"
  "matrix is assumed.\n"},
{"threshold", FX_PARAM, FX_DOUBLE, NULL,
  "The threshold for cutting off a shell-pair.  Default: 10e-10\n"},
/*{"centers_out", FX_PARAM, FX_STR, NULL,
  "The file to write the charge centers to.  Default: centers.csv \n"},
{"exponents_out", FX_PARAM, FX_STR, NULL, 
  "The file to write the charge exponents to.  Default: exp.csv\n"}*/
};

const fx_module_doc fock_matrix_main_doc = {
  cfmm_screening_entries, NULL, 
  "Runs and compares different fock matrix construction methods.\n"
};



int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, &fock_matrix_main_doc);
  
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
  
  
  
  
  

  return 0;

} // int main()