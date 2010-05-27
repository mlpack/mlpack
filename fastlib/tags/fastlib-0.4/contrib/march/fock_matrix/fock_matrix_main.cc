#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/multi_tree/multi_tree_fock.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"
#include "contrib/march/fock_matrix/link/link.h"
#include "contrib/march/fock_matrix/cfmm/cfmm_coulomb.h"
#include "fock_matrix_comparison.h"
#include "chem_reader/chem_reader.h"


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
{"nuclear_centers", FX_PARAM, FX_STR, NULL,
  "The centers of the nuclei for comparing the final energies.  If not specified\n"
"then the algorithm will not compare energies.\n"},
{"nuclear_charges", FX_PARAM, FX_STR, NULL,
  "The charges of the nuclei.  If not specified, all atoms are assumed to be H.\n"},
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
  "Compare the result to prescreening. \n"}, 
{"compare_link", FX_PARAM, FX_STR, NULL,
  "Compare the result to prescreening. \n"}, 
{"compare_multi", FX_PARAM, FX_STR, NULL,
  "Compare the result to prescreening. \n"}, 
{"compare_naive", FX_PARAM, FX_STR, NULL,
  "Compare the naive result to prescreening.\n"},
{"prescreening_storage", FX_PARAM, FX_STR, NULL,
  "Location of saved prescreening results.\n"},
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
  
  eri::ERIInit();
  
  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);

  if (centers.n_cols() != exp_mat.n_cols()) {
    FATAL("Number of basis centers must equal number of exponents.\n");
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
  
  
  std::string density_str;
  Matrix density;
  // WARNING: this hack only works for s and p functions
  int num_functions = centers.n_cols() + 2*(int)la::Dot(momenta, momenta);
  
  if (fx_param_exists(root_mod, "density")) {
 
    const char* density_file = fx_param_str_req(root_mod, "density");
    density_str = density_file;
    
    size_t density_ext = density_str.find_last_of(".");
    if (!strcmp("qcmat", 
                density_str.substr(density_ext+1, std::string::npos).c_str())) {
     
      printf("Reading QChem Style Density Matrix.\n");
      chem_reader::ReadQChemDensity(density_file, &density, num_functions);
      // QC density matrices are only for alpha electrons
      la::Scale(2.0, &density);
      
    }
    else {
      
      printf("Reading csv Density Matrix.\n");
      data::Load(density_file, &density);
      
    }
  }
  else {
    density.Init(num_functions, num_functions);
    density.SetAll(1.0);
    density_str = "default";
  }
  
  //density.PrintDebug("Density (from input file)");
  /* NO LONGER TRUE
  if ((density.n_cols() != centers.n_cols()) || 
      (density.n_rows() != centers.n_cols())) {
    FATAL("Density matrix has wrong dimensions.\n");
  }
   */
  
    
  Matrix nuclear_centers_mat;
  Matrix* nuclear_centers;
  if (fx_param_exists(root_mod, "nuclear_centers")) {
    const char* nuclear_centers_file = fx_param_str_req(root_mod, 
                                                        "nuclear_centers");
    data::Load(nuclear_centers_file, &nuclear_centers_mat);
    nuclear_centers = &nuclear_centers_mat;
  }
  else {
    nuclear_centers_mat.Init(1,1);
    nuclear_centers = NULL; 
  }
  
  Matrix nuclear_charges;
  if (fx_param_exists(root_mod, "nuclear_charges")) {
    const char* nuclear_charges_file = fx_param_str_req(root_mod, 
                                                        "nuclear_charges");
    data::Load(nuclear_charges_file, &nuclear_charges);
  }
  else if (nuclear_centers) {
    nuclear_charges.Init(1, nuclear_centers->n_cols());
    nuclear_charges.SetAll(1.0);
    printf("Assuming all H atoms.\n\n");
  }
  else {
    nuclear_charges.Init(1,1); 
  }
  
  if (nuclear_centers) {
    if (nuclear_centers->n_cols() != nuclear_charges.n_cols()) {
      FATAL("Must provide a charge for every nuclear center!\n");
    }
  }
  
  const double angstrom_to_bohr = 1.889725989;
  // if the data are not input in bohr, assume they are in angstroms
  if (!fx_param_exists(root_mod, "bohr")) {
    
    la::Scale(angstrom_to_bohr, &centers);
    if (nuclear_centers) {
      la::Scale(angstrom_to_bohr, nuclear_centers);
    }
  }
  
  // Have the naive matrices on hand if needed 
  // maybe turn these into pointers if so I can see if they're filled
  Matrix prescreening_fock;
  Matrix prescreening_coulomb;
  Matrix prescreening_exchange;
  
  // these won't work with fx_run
  const char* dir_char = fx_param_str(root_mod, "prescreening_storage", 
                                      "prescreening_storage/");
                                      
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
  
  size_t density_last = density_str.find_last_of("/");
  if (density_last != std::string::npos) {
    density_str.erase(0, density_last+1);
  }
  const char* density_name = density_str.c_str();
  
  std::string directory(dir_char);
  std::string underscore("_");
  std::string under_F("_F.csv");
  std::string under_J("_J.csv");
  std::string under_K("_K.csv");
  std::string core_name("core.csv");
  std::string change_name("change.csv");
  std::string thresh_str;
  
  double prescreening_threshold = fx_param_double(root_mod, 
                                                  "prescreening/thresh", 
                                                  10e-10);
  
  // this stopped compiling for some reason - needs to be fixed
  //std::ostringstream oss;
  //oss << prescreening_threshold;
  //thresh_str = oss.str();
  
  std::string prescreening_fock_string;
  //prescreening_fock_string = directory + centers_name + underscore + exp_name 
  //                            + underscore + density_name + underscore  
  //                            + thresh_str + under_F;
  prescreening_fock_string = directory + centers_name + underscore + exp_name 
  + underscore + density_name + underscore  
  + under_F;
  const char* prescreening_fock_file = prescreening_fock_string.c_str();

  std::string prescreening_coulomb_string;
  //prescreening_coulomb_string = directory + centers_name + underscore + exp_name 
  //                        + underscore + density_name + underscore  
  //                        + thresh_str + under_J;
  prescreening_coulomb_string = directory + centers_name + underscore + exp_name 
  + underscore + density_name + underscore  
  + under_J;
  const char* prescreening_coulomb_file = prescreening_coulomb_string.c_str();

  std::string prescreening_exchange_string;
  //prescreening_exchange_string = directory + centers_name + underscore + exp_name
  //                        + underscore + density_name + underscore  
  //                        + thresh_str + under_K;
  prescreening_exchange_string = directory + centers_name + underscore + exp_name
  + underscore + density_name + underscore  
  + under_K;
  const char* prescreening_exchange_file = prescreening_exchange_string.c_str();
  
  
  std::string core_string;
  core_string = directory + centers_name + underscore + exp_name + underscore 
                + core_name;
  const char* core_file = core_string.c_str();
  
  std::string change_string;
  change_string = directory + centers_name + underscore + exp_name + underscore 
                  + change_name;
  const char* change_file = change_string.c_str();

  
  bool do_prescreening = fx_param_exists(root_mod, "do_prescreening");
  
  fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
  
  Matrix** prescreening_mats;
  
  // if we are going to compare
  if (!do_prescreening) {
    if (fx_param_exists(root_mod, "compare_cfmm") || 
        fx_param_exists(root_mod, "compare_link") || 
        fx_param_exists(root_mod, "compare_naive") || 
        fx_param_exists(root_mod, "compare_multi")) {


      // try to load them
      // assuming that all will load or none will
      if (data::Load(prescreening_fock_file, &prescreening_fock) == SUCCESS_FAIL) {
       
        // destruct them if they didn't load
        prescreening_fock.Destruct();
        
        // if it's not already going to get done, it needs to be done
        do_prescreening = true;
        
      }
      else {
      
        data::Load(prescreening_coulomb_file, &prescreening_coulomb);
        data::Load(prescreening_exchange_file, &prescreening_exchange);
      
      }
    
    }
    else {
      prescreening_fock.Init(1,1);
      prescreening_coulomb.Init(1,1);
      prescreening_exchange.Init(1,1);
    }
  }
    
  // I think this fails if the files exist but --do_naive is specified
  if (do_prescreening) {
    
    printf("======== Prescreening Computation ========\n");
    
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
    
    data::Save(prescreening_fock_file, prescreening_fock);
    data::Save(prescreening_coulomb_file, prescreening_coulomb);
    data::Save(prescreening_exchange_file, prescreening_exchange);
    
  } // do_prescreening
  
  prescreening_mats = (Matrix**)malloc(3*sizeof(Matrix*));
  prescreening_mats[0] = &prescreening_fock;
  prescreening_mats[1] = &prescreening_coulomb;
  prescreening_mats[2] = &prescreening_exchange;
  
  
  Matrix cfmm_coulomb;
  Matrix link_exchange;
  
  
  if (fx_param_exists(root_mod, "do_cfmm") 
      && fx_param_exists(root_mod, "do_link")) {
    
    printf("======== CFMM Computation ========\n");
    
    
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    
    CFMMCoulomb coulomb_alg;
    
    coulomb_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);
    coulomb_alg.Compute();
    coulomb_alg.OutputCoulomb(&cfmm_coulomb);
    
    coulomb_alg.Destruct();
    
    if (fx_param_exists(root_mod, "print_cfmm")) {
      cfmm_coulomb.PrintDebug("CFMM J");
    }
    
    printf("======== LinK Computation ========\n");
    
    
    fx_module* link_mod = fx_submodule(root_mod, "link");
    
    Link link_alg;
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);
    link_alg.Compute();
    link_alg.OutputExchange(&link_exchange);
    
    link_alg.Destruct();
    
    if (fx_param_exists(root_mod, "print_link")) {
      
      link_exchange.PrintDebug("LinK K");
      
    }
    
    
    
    fx_module* cfmm_compare_mod = fx_submodule(cfmm_mod, "compare");
    
    Matrix cfmm_link_fock;
    la::SubInit(link_exchange, cfmm_coulomb, &cfmm_link_fock);
    
    Matrix** cfmm_mats;
    cfmm_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
    cfmm_mats[0] = &cfmm_link_fock;
    cfmm_mats[1] = &cfmm_coulomb;
    cfmm_mats[2] = &link_exchange;
    
    FockMatrixComparison cfmm_compare;
    cfmm_compare.Init(cfmm_mod, cfmm_mats, prescreening_mod, prescreening_mats, 
                      centers, exp_mat, momenta, density, nuclear_centers,
                      nuclear_charges, cfmm_compare_mod, core_file, change_file);
    cfmm_compare.Compare();
    
    
    
  }
  else if (fx_param_exists(root_mod, "do_cfmm")) {
  
    printf("======== CFMM Computation ========\n");
  
    
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
      cfmm_compare.Init(cfmm_mod, cfmm_mats, prescreening_mod, prescreening_mats, 
                        centers, exp_mat, momenta, density, nuclear_centers,
                        nuclear_charges, cfmm_compare_mod, core_file, change_file);
      cfmm_compare.Compare();
    
    } // cfmm comparison
    
    link_exchange.Init(1,1);
        
  } // do_cfmm
  else if (fx_param_exists(root_mod, "do_link")) {
    
    printf("======== LinK Computation ========\n");
    
    
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
      link_compare.Init(link_mod, link_mats, prescreening_mod, prescreening_mats, 
                        centers, exp_mat, momenta, density, nuclear_centers,
                        nuclear_charges, link_compare_mod, core_file, change_file);
      link_compare.Compare();
      
    } // link comparison
        
    cfmm_coulomb.Init(1,1);
    
  } // do_link
  else {
    
    cfmm_coulomb.Init(1,1);
    link_exchange.Init(1,1);
    
  }
   

  if (fx_param_exists(root_mod, "do_naive")) {
    
    printf("======== Naive Computation ========\n");
    
    Matrix naive_fock;
    Matrix naive_coulomb;
    Matrix naive_exchange;
    
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    
    NaiveFockMatrix naive_alg;
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);
    
    naive_alg.Compute();
    naive_alg.OutputFock(&naive_fock, &naive_coulomb, 
                             &naive_exchange);
    
    if (fx_param_exists(root_mod, "print_naive")) {
      
      naive_fock.PrintDebug("Naive F");
      naive_coulomb.PrintDebug("Naive J");
      naive_exchange.PrintDebug("Naive K");
      
    }
    
    
    if (fx_param_exists(root_mod, "compare_naive")) {
      
      fx_module* naive_compare_mod = fx_submodule(naive_mod, "compare");
      
      Matrix** naive_mats;
      naive_mats = (Matrix**)malloc(3 * sizeof(Matrix*));
      naive_mats[0] = &naive_fock;
      naive_mats[1] = &naive_coulomb;
      naive_mats[2] = &naive_exchange;
      
      FockMatrixComparison naive_compare;
      naive_compare.Init(naive_mod, naive_mats, prescreening_mod, 
                         prescreening_mats, centers, exp_mat, momenta, 
                         density, nuclear_centers, nuclear_charges, 
                         naive_compare_mod, core_file, change_file);
      naive_compare.Compare();
      
    } // Naive comparison
        
    
  } // do_naive
  
    
  
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
      
      multi_alg.Destruct();
      
      FockMatrixComparison multi_compare;
      multi_compare.Init(multi_mod, multi_mats, prescreening_mod, 
                         prescreening_mats, centers, exp_mat, momenta, density, 
                         nuclear_centers, nuclear_charges, multi_compare_mod,
                         core_file, change_file);
      multi_compare.Compare();
      
    } // cfmm comparison        
    
  } // do_multi
  

  eri::ERIFree();
  
  fx_done(root_mod);

  return 0;

} // int main()