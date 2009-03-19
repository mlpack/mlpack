#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/multi_tree/multi_tree_fock.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"
#include "contrib/march/fock_matrix/link/link.h"
#include "contrib/march/fock_matrix/cfmm/cfmm_coulomb.h"


const fx_entry_doc fock_matrix_main_entries[] = {
{"centers", FX_REQUIRED, FX_STR, NULL, 
  "A file containing the centers of the basis functions.\n"},
{"bohr", FX_PARAM, FX_STR, NULL, 
  "Specify this parameter if the data are in bohr.  Otherwise they are assumed\n"
  " to be in angstroms.\n"},
{"exponents", FX_REQUIRED, FX_STR, NULL, 
  "A file containing the exponents of the basis functions.\n"
  "Must have the same number of rows as centers.\n"},
{"density", FX_PARAM, FX_STR, NULL, 
  "A file containing the density matrix.  If it is not provided, an all-ones\n"
  "matrix is assumed.\n"},
{"momenta", FX_PARAM, FX_STR, NULL, 
"A file containing the momenta.  If not specified, then all functions are\n"
"assumed to be s-type.\n"},
{"do_cfmm", FX_PARAM, FX_STR, NULL,
  "Compute the CFMM Coulomb matrix.  The value is irrelevant.\n"},
{"do_link", FX_PARAM, FX_STR, NULL,
  "Compute the LinK exchange matrix.  The value is irrelevant.\n"},
{"do_prescreening", FX_PARAM, FX_STR, NULL,
  "Compute the Fock matrix with Scwartz prescreening.  The value is irrelevant.\n"},
{"do_naive", FX_PARAM, FX_STR, NULL,
  "Compute the Fock matrix naively.  The value is irrelevant.\n"},
{"do_multi", FX_PARAM, FX_STR, NULL,
  "Compute the multi-tree Fock matrix.  The value is irrelevant.\n"},
{"print_cfmm", FX_PARAM, FX_STR, NULL,
  "Print the CFMM Coulomb matrix.  The value is irrelevant.\n"},
{"print_link", FX_PARAM, FX_STR, NULL,
  "Print the LinK exchange matrix.  The value is irrelevant.\n"},
{"print_prescreening", FX_PARAM, FX_STR, NULL,
  "Print the Fock matrix with Scwartz prescreening.  The value is irrelevant.\n"},
{"print_naive", FX_PARAM, FX_STR, NULL,
  "Print the Fock matrix naively.  The value is irrelevant.\n"},
{"print_multi", FX_PARAM, FX_STR, NULL,
  "Print the multi-tree Fock matrix.  The value is irrelevant.\n"},  
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc fock_matrix_main_submodules[] = {
  {"cfmm", &cfmm_mod_doc, 
   "Parameters and results for the CFMM.\n"},
  {"link", &link_mod_doc,
   "Parameters and results for LinK.\n"},
  {"prescreening", &prescreening_mod_doc,
   "Parameters and results for Schwartz prescreening.\n"},
  {"naive", &naive_mod_doc,
   "Parameters and results for naive.\n"},
  {"multi", &multi_mod_doc,
   "Parameters and results for multi-tree algorithm.\n"},
  FX_SUBMODULE_DOC_DONE
};


const fx_module_doc fock_matrix_main_doc = {
  fock_matrix_main_entries, fock_matrix_main_submodules, 
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
    data::Load(density_file, &density);
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
    data::Load(momenta_file, &momenta);
  }
  else {
    momenta.Init(1, centers.n_cols());
    momenta.SetAll(0);
  }
  
  const double angstrom_to_bohr = 1.889725989;
  // if the data are not input in bohr, assume they are in angstroms
  if (!fx_param_exists(root_mod, "bohr")) {
    
    la::Scale(angstrom_to_bohr, &centers);
  
  }
  
  
  if (fx_param_exists(root_mod, "do_cfmm")) {
  
    Matrix cfmm_coulomb;
 
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    
    CFMMCoulomb coulomb_alg;
    
    coulomb_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);
    coulomb_alg.ComputeCoulomb();
    coulomb_alg.Output(&cfmm_coulomb);
  
    if (fx_param_exists(root_mod, "print_cfmm")) {
      cfmm_coulomb.PrintDebug("CFMM J");
    }
        
  } // do_cfmm
  


  if (fx_param_exists(root_mod, "do_link")) {
    
    Matrix link_exchange;
    
    fx_module* link_mod = fx_submodule(root_mod, "link");
    
    Link link_alg;
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);
    link_alg.ComputeExchangeMatrix();
    link_alg.OutputExchangeMatrix(&link_exchange);
    
    if (fx_param_exists(root_mod, "print_link")) {
      
      link_exchange.PrintDebug("LinK K");
      
    }
    
  } // do_link


  if (fx_param_exists(root_mod, "do_prescreening")) {
    
    Matrix prescreening_fock;
    
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    
    SchwartzPrescreening prescreen_alg;
    prescreen_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    prescreen_alg.ComputeFockMatrix(&prescreening_fock);
    
    if (fx_param_exists(root_mod, "print_prescreening")) {
      
      prescreening_fock.PrintDebug("Schwartz Prescreening F");
      
    }
    
  } // do_prescreening
  
  if (fx_param_exists(root_mod, "do_naive")) {
    
    Matrix naive_fock;
    Matrix naive_coulomb;
    Matrix naive_exchange;
    
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    
    NaiveFockMatrix naive_alg;
    
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);
    naive_alg.ComputeFock();
    naive_alg.OutputFock(&naive_fock, &naive_coulomb, &naive_exchange);
    
    if (fx_param_exists(root_mod, "print_naive")) {
      
      naive_fock.PrintDebug("Naive F");
      naive_coulomb.PrintDebug("Naive J");
      naive_exchange.PrintDebug("Naive K");
      
    }
    
  } // do_naive
  

  if (fx_param_exists(root_mod, "do_multi")) {
    
    Matrix multi_fock;
    Matrix multi_coulomb;
    Matrix multi_exchange;
    
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    
    MultiTreeFock multi_alg;
    
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);
    multi_alg.ComputeFockMatrix();
    multi_alg.OutputFockMatrix(&multi_fock, &multi_coulomb, &multi_exchange, 
                               NULL);
    
    if (fx_param_exists(root_mod, "print_multi")) {
      
      multi_fock.PrintDebug("Multi F");
      multi_coulomb.PrintDebug("Multi J");
      multi_exchange.PrintDebug("Multi K");
      
    }
    
  } // do_multi

  
  // Do comparison here?
  
  if (fx_param_exists(root_mod, "compare")) {
  
    
  
  } // comparison
  
  fx_done(root_mod);

  return 0;

} // int main()