#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/cfmm/cfmm_coulomb.h"
#include "contrib/march/fock_matrix/multi_tree/multi_tree_fock.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"
#include "contrib/march/fock_matrix/link/link.h"


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
  
  Matrix density;
  if (fx_param_exists(root_mod, "density")) {
    
    const char* density_file = fx_param_str_req(root_mod, "density");
    data::Load(density_file, &root_mod);
    
  }
  else {
    
    density.Init(centers.n_cols(), centers.n_cols());
    density.SetAll(1.0);
    
  }
  
  if ((density.n_cols() != centers.n_cols()) || 
      (density.n_rows() != centers.n_cols())) {
    FATAL("Density matrix has wrong dimensions.\n");
  }
  
  Matrix momenta;
  if (fx_param_exists(root_mod, "momenta")) {
    const char* momenta_file = fx_param_str_req(root_mod, "momenta");
  }
  else {
    momenta.Init(1, centers.n_cols());
    momenta.SetAll(0);
  }
  
  
  if (fx_param_exists(root_mod, "do_cfmm")) {
  
    if (fx_param_exists(root_mod, "print_cfmm")) {
    
    }
  
  } // do_cfmm


  if (fx_param_exists(root_mod, "do_link")) {
    
    if (fx_param_exists(root_mod, "print_link")) {
      
    }
    
  } // do_link

  if (fx_param_exists(root_mod, "do_prescreening")) {
    
    if (fx_param_exists(root_mod, "print_prescreening")) {
      
    }
    
  } // do_prescreening
  
  if (fx_param_exists(root_mod, "do_naive")) {
    
    if (fx_param_exists(root_mod, "print_naive")) {
      
    }
    
  } // do_naive
  
  if (fx_param_exists(root_mod, "do_multi")) {
    
    if (fx_param_exists(root_mod, "print_multi")) {
      
    }
    
  } // do_multi
  
  
  
  

  return 0;

} // int main()