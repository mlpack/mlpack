#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/multi_tree/multi_tree_fock.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"
#include "contrib/march/fock_matrix/link/link.h"
#include "contrib/march/fock_matrix/cfmm/cfmm_coulomb.h"
#include "fock_matrix_comparison.h"


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
  "Compute the Fock matrix naively.  Specifying this will recompute the naive\n"
  "matrices, even if they already exist.\n"},
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
{"compare_cfmm", FX_PARAM, FX_STR, NULL,
  "Compare the result to naive. \n"}, 
{"compare_link", FX_PARAM, FX_STR, NULL,
  "Compare the result to naive. \n"}, 
{"compare_prescreening", FX_PARAM, FX_STR, NULL,
  "Compare the result to naive. \n"}, 
{"compare_multi", FX_PARAM, FX_STR, NULL,
  "Compare the result to naive. \n"}, 
{"naive_storage", FX_PARAM, FX_STR, NULL,
  "Location of saved naive results.\n"},
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
  
  // Have the naive matrices on hand if needed 
  // maybe turn these into pointers if so I can see if they're filled
  Matrix naive_fock;
  Matrix naive_coulomb;
  Matrix naive_exchange;
  
  // these won't work with fx_run
  const char* dir_char = fx_param_str(root_mod, "naive_storage", 
                                      "naive_storage/");
                                      
  // strip the filenames for use with fx-run
  std::string centers_string(centers_file);
  size_t centers_last = centers_string.find_last_of("/");
  if (centers_last != std::string::npos) {
    centers_string.erase(0, centers_last+1);
  } 
  const char* centers_name = centers_string.c_str();
  
  std::string exp_string(exp_file);
  size_t exp_last = exp_string.find_last_of("/");
  if (exp_last != std::string::npos) {
    exp_string.erase(0, exp_last+1);
  }
  const char* exp_name = exp_string.c_str();
  
  /*
  printf(centers_name);
  printf("\n");
  printf(exp_name);
  printf("\n\n");
  */
  
  std::string directory(dir_char);
  std::string underscore("_");
  std::string under_F("_F.csv");
  std::string under_J("_J.csv");
  std::string under_K("_K.csv");
  
  std::string naive_fock_string;
  naive_fock_string = directory + centers_name + underscore + exp_name + under_F;
  const char* naive_fock_file = naive_fock_string.c_str();

  std::string naive_coulomb_string;
  naive_coulomb_string = directory + centers_name + underscore + exp_name + under_J;
  const char* naive_coulomb_file = naive_coulomb_string.c_str();

  std::string naive_exchange_string;
  naive_exchange_string = directory + centers_name + underscore + exp_name + under_K;
  const char* naive_exchange_file = naive_exchange_string.c_str();

  
  bool do_naive = fx_param_exists(root_mod, "do_naive");
  
  fx_module* naive_mod = fx_submodule(root_mod, "naive");
  
  Matrix** naive_mats;
  
  // if we are going to compare
  if (!do_naive) {
    if (fx_param_exists(root_mod, "compare_cfmm") || 
        fx_param_exists(root_mod, "compare_link") || 
        fx_param_exists(root_mod, "compare_prescreening") || 
        fx_param_exists(root_mod, "compare_multi")) {


      // try to load them
      // assuming that all will load or none will
      if (data::Load(naive_fock_file, &naive_fock) == SUCCESS_FAIL) {
       
        // destruct them if they didn't load
        naive_fock.Destruct();
        
        // if it's not already going to get done, it needs to be done
        do_naive = true;
        
      }
      else {
      
        data::Load(naive_coulomb_file, &naive_coulomb);
        data::Load(naive_exchange_file, &naive_exchange);
      
      }
    
    }
    else {
      naive_fock.Init(1,1);
      naive_coulomb.Init(1,1);
      naive_exchange.Init(1,1);
    }
  }
    
  // I think this fails if the files exist but --do_naive is specified
  if (do_naive) {
    
    printf("======== Naive Computation ========\n");
    
    NaiveFockMatrix naive_alg;
    
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);
    naive_alg.Compute();
    naive_alg.OutputFock(&naive_fock, &naive_coulomb, &naive_exchange);
    
    if (fx_param_exists(root_mod, "print_naive")) {
      
      naive_fock.PrintDebug("Naive F");
      naive_coulomb.PrintDebug("Naive J");
      naive_exchange.PrintDebug("Naive K");
      
    }
    
    data::Save(naive_fock_file, naive_fock);
    data::Save(naive_coulomb_file, naive_coulomb);
    data::Save(naive_exchange_file, naive_exchange);
    
  } // do_naive
  
  naive_mats = (Matrix**)malloc(3*sizeof(Matrix*));
  naive_mats[0] = &naive_fock;
  naive_mats[1] = &naive_coulomb;
  naive_mats[2] = &naive_exchange;
  
  
  if (fx_param_exists(root_mod, "do_cfmm")) {
  
    printf("======== CFMM Computation ========\n");
  
    Matrix cfmm_coulomb;
 
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    
    CFMMCoulomb coulomb_alg;
    
    coulomb_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);
    coulomb_alg.Compute();
    coulomb_alg.OutputCoulomb(&cfmm_coulomb);
  
    if (fx_param_exists(root_mod, "print_cfmm")) {
      cfmm_coulomb.PrintDebug("CFMM J");
    }
    
    if (fx_param_exists(root_mod, "compare_cfmm")) {
    
      fx_module* cfmm_compare_mod = fx_submodule(cfmm_mod, "compare");
      
      Matrix** cfmm_mats;
      cfmm_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
      cfmm_mats[0] = NULL;
      cfmm_mats[1] = &cfmm_coulomb;
      cfmm_mats[2] = NULL;
      
      FockMatrixComparison cfmm_compare;
      cfmm_compare.Init(cfmm_mod, cfmm_mats, naive_mod, naive_mats, 
                        cfmm_compare_mod);
      cfmm_compare.Compare();
    
    } // cfmm comparison
        
  } // do_cfmm
  


  if (fx_param_exists(root_mod, "do_link")) {
    
    printf("======== LinK Computation ========\n");
    
    Matrix link_exchange;
    
    fx_module* link_mod = fx_submodule(root_mod, "link");
    
    Link link_alg;
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);
    link_alg.Compute();
    link_alg.OutputExchange(&link_exchange);
    
    if (fx_param_exists(root_mod, "print_link")) {
      
      link_exchange.PrintDebug("LinK K");
      
    }
    
    if (fx_param_exists(root_mod, "compare_link")) {
      
      fx_module* link_compare_mod = fx_submodule(link_mod, "compare");
      
      Matrix** link_mats;
      link_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
      link_mats[0] = NULL;
      link_mats[1] = NULL;
      link_mats[2] = &link_exchange;
      
      FockMatrixComparison link_compare;
      link_compare.Init(link_mod, link_mats, naive_mod, naive_mats, 
                        link_compare_mod);
      link_compare.Compare();
      
    } // cfmm comparison
        
    
  } // do_link


  if (fx_param_exists(root_mod, "do_prescreening")) {
    
    printf("======== Prescreening Computation ========\n");
    
    Matrix prescreening_fock;
    Matrix prescreening_coulomb;
    Matrix prescreening_exchange;
    
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    
    SchwartzPrescreening prescreen_alg;
    prescreen_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    prescreen_alg.Compute();
    prescreen_alg.OutputFock(&prescreening_fock, &prescreening_coulomb, 
                             &prescreening_exchange);
    
    if (fx_param_exists(root_mod, "print_prescreening")) {
      
      prescreening_fock.PrintDebug("Schwartz Prescreening F");
      prescreening_coulomb.PrintDebug("Schwartz Prescreening J");
      prescreening_exchange.PrintDebug("Schwartz Prescreening K");
      
    }
    
    if (fx_param_exists(root_mod, "compare_prescreening")) {
      
      fx_module* prescreening_compare_mod = fx_submodule(prescreening_mod, "compare");
      
      Matrix** prescreening_mats;
      prescreening_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
      prescreening_mats[0] = &prescreening_fock;
      prescreening_mats[1] = &prescreening_coulomb;
      prescreening_mats[2] = &prescreening_exchange;
      
      FockMatrixComparison prescreening_compare;
      prescreening_compare.Init(prescreening_mod, prescreening_mats, naive_mod, 
                                naive_mats, prescreening_compare_mod);
      prescreening_compare.Compare();
      
    } // cfmm comparison
        
    
  } // do_prescreening
  
    

  if (fx_param_exists(root_mod, "do_multi")) {
    
    printf("======== Multi-Tree Computation ========\n");
    
    Matrix multi_fock;
    Matrix multi_coulomb;
    Matrix multi_exchange;
    
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    
    MultiTreeFock multi_alg;
    
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);
    multi_alg.Compute();
    multi_alg.OutputFockMatrix(&multi_fock, &multi_coulomb, &multi_exchange, 
                               NULL);
    
    if (fx_param_exists(root_mod, "print_multi")) {
      
      multi_fock.PrintDebug("Multi F");
      multi_coulomb.PrintDebug("Multi J");
      multi_exchange.PrintDebug("Multi K");
      
    }
    
    if (fx_param_exists(root_mod, "compare_multi")) {
      
      fx_module* multi_compare_mod = fx_submodule(multi_mod, "compare");
      
      Matrix** multi_mats;
      multi_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
      multi_mats[0] = &multi_fock;
      multi_mats[1] = &multi_coulomb;
      multi_mats[2] = &multi_exchange;
      
      FockMatrixComparison multi_compare;
      multi_compare.Init(multi_mod, multi_mats, naive_mod, 
                                naive_mats, multi_compare_mod);
      multi_compare.Compare();
      
    } // cfmm comparison        
    
  } // do_multi



  fx_done(root_mod);

  return 0;

} // int main()