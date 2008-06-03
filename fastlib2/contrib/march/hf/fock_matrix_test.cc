#include "dual_tree_integrals.h"
#include "naive_fock_matrix.h"
//#include <fastlib/base/test.h>

class FockMatrixTest {
  
public:
  
  void Setup(const char* centers_name, const char* density_name, 
             struct datanode* multi_mod, struct datanode* naive_mod, 
             double band) {
    
    Matrix test_centers;
    data::Load(centers_name, &test_centers);
    
    Matrix test_density;
    data::Load(density_name, &test_density);
        
    Matrix test_core;
    test_core.Init(test_centers.n_cols(), test_centers.n_cols());
    test_core.SetZero();
    
    multi_ = new DualTreeIntegrals();
    multi_->Init(test_centers, multi_mod, band);
    multi_->SetDensity(test_density);
    
    naive_ = new NaiveFockMatrix();
    naive_->Init(test_centers, naive_mod, test_density, test_core, band);
                     
  } // Setup
  
  
  void Destruct() { 
    
    old_from_new.Destruct();
    delete multi_;
    delete naive_;
    
  } // Destruct
  
  void CompareMatrices(struct datanode* mod) {
  
    double coulomb_error = 0.0;
    double max_coulomb_error = 0.0;
    double exchange_error = 0.0;
    double max_exchange_error = 0.0;
    
    double rel_coulomb_error = 0.0;
    double max_rel_coulomb_error = 0.0;
    double rel_exchange_error = 0.0;
    double max_rel_exchange_error = 0.0;
    
    double min_coulomb = DBL_INF;
    double max_coulomb = -DBL_INF;
    double min_exchange = DBL_INF;
    double max_exchange = -DBL_INF;
    
    Matrix multi_exchange;
    multi_exchange.Alias(multi_->exchange_matrix_);
    Matrix multi_coulomb;
    multi_coulomb.Alias(multi_->coulomb_matrix_);
    Matrix naive_exchange;
    naive_exchange.Alias(naive_->exchange_matrix_);
    Matrix naive_coulomb;
    naive_coulomb.Alias(naive_->coulomb_matrix_);
    
    /*printf("MULTI:\n");
    multi_exchange.PrintDebug();
    printf("NAIVE:\n");
    naive_exchange.PrintDebug();
    printf("\n\n");
    */
    
    index_t num_rows = multi_exchange.n_rows();
    //printf("num_rows: %d\n", num_rows);
    
    for (index_t i = 0; i < num_rows; i++) {
      
      for (index_t j = 0; j < num_rows; j++) {
        
        double this_val;
        double this_naive;
        this_naive = fabs(naive_exchange.get(old_from_new[i], old_from_new[j]));
        //printf("i:%d, j:%d, this_val:%g\n", i, j, this_val);
        if (this_naive > max_exchange) {
          max_exchange = this_naive;
        }
        if (this_naive < min_exchange) {
          min_exchange = this_naive;
        }
        
        this_val = fabs(multi_exchange.get(i, j) - 
                        naive_exchange.get(old_from_new[i], old_from_new[j]));
        exchange_error = exchange_error + this_val;
        
        if (this_val > max_exchange_error) { 
          max_exchange_error = this_val;
          //printf("max_error: (%d, %d)\n", i, j);
        } 
        
        this_val = this_val/this_naive;
        rel_exchange_error = rel_exchange_error + this_val;
        
        if (this_val > max_rel_exchange_error) {
          max_rel_exchange_error = this_val;
        }
        
        //////////// Coulomb //////////////
        
        this_naive = fabs(naive_coulomb.get(old_from_new[i], old_from_new[j]));
        this_val = fabs(naive_coulomb.get(old_from_new[i], old_from_new[j])
                        - multi_coulomb.get(i, j));
        
        if (this_naive > max_coulomb) {
          max_coulomb = this_naive;
        }
        if (this_naive < min_coulomb) {
          min_coulomb = this_naive;
        }
        
        coulomb_error = coulomb_error + this_val;
        
        if (this_val > max_coulomb_error) { 
          max_coulomb_error = this_val;
        }     
        
        this_val = this_val/this_naive;
        rel_coulomb_error = rel_coulomb_error + this_val;
        
        if (this_val > max_rel_coulomb_error) {
          max_rel_coulomb_error = this_val;
        }
        
      } // j
      
    } // i 
    
    exchange_error = exchange_error/(num_rows * num_rows);
    coulomb_error = coulomb_error/(num_rows * num_rows);
        
    rel_exchange_error = rel_exchange_error/(num_rows * num_rows);
    rel_coulomb_error = rel_coulomb_error/(num_rows * num_rows);
    
    fx_format_result(mod, "max_coulomb", "%g", max_coulomb);
    fx_format_result(mod, "min_coulomb", "%g", min_coulomb);
    fx_format_result(mod, "max_exchange", "%g", max_exchange);
    fx_format_result(mod, "min_exchange", "%g", min_exchange);
    
    
    fx_format_result(mod, "ave_abs_coulomb_error", "%g", coulomb_error);
    fx_format_result(mod, "max_abs_coulomb_error", "%g", max_coulomb_error);
    
    fx_format_result(mod, "ave_abs_exchange_error", "%g", exchange_error);
    fx_format_result(mod, "max_abs_exchange_error", "%g", max_exchange_error);
    
    fx_format_result(mod, "ave_rel_coulomb_error", "%g", rel_coulomb_error);
    fx_format_result(mod, "max_rel_coulomb_error", "%g", max_rel_coulomb_error);
    
    fx_format_result(mod, "ave_rel_exchange_error", "%g", rel_exchange_error);
    fx_format_result(mod, "max_rel_exchange_error", "%g", 
                     max_rel_exchange_error);
    
  } // CompareMatrices()

  void TestMatricesSmall() {
  
  
    struct datanode* multi_mod_small = fx_submodule(NULL, "multi_small", 
                                                    "multi_small");
                                                    
    fx_set_param(multi_mod_small, "epsilon", "0.0");
    
    struct datanode* naive_mod_small = fx_submodule(NULL, "naive_small", 
                                                    "naive_small");
                                                    
    fx_set_param(naive_mod_small, "coulomb_output", "naive_coulomb_4_1.0.csv");
    fx_set_param(naive_mod_small, "exchange_output", 
                 "naive_exchange_4_1.0.csv");
    
    Setup("test_centers.csv", "test_density.csv", 
          multi_mod_small, naive_mod_small, 1.0);
    
    multi_->ComputeFockMatrix();
    Matrix cou;
    Matrix exc;
    multi_->OutputFockMatrix(NULL, &cou, &exc, &old_from_new);
    
    naive_->ComputeFockMatrix();
    
    CompareMatrices(multi_mod_small);
    
        
    Destruct();
  
  } // TestMatricesSmall() 
  
  
  void TestMatricesMidNoPrune() {
  
    struct datanode* multi_mod_noprune = fx_submodule(NULL, "multi_noprune", 
                                                      "multi_noprune");
    
    fx_set_param(multi_mod_noprune, "epsilon", "0.0");
    fx_set_param(multi_mod_noprune, "hybrid_cutoff", "0.0");
    fx_set_param(multi_mod_noprune, "epsilon_absolute", "0.0");
    fx_set_param(multi_mod_noprune, "leaf_size", "1");
    
    struct datanode* naive_mod_noprune = fx_submodule(NULL, "naive_noprune", 
                                                      "naive_noprune");
    
    fx_set_param(naive_mod_noprune, "coulomb_output", "naive_coulomb_10_1.0.csv");
    fx_set_param(naive_mod_noprune, "exchange_output", 
                 "naive_exchange_10_1.0.csv");
    
    Setup("test_centers_10.csv", "test_density_10.csv", 
          multi_mod_noprune, naive_mod_noprune, 1.0);
    
    multi_->ComputeFockMatrix();
    Matrix cou;
    Matrix exc;
    multi_->OutputFockMatrix(NULL, &cou, &exc, &old_from_new);
    
    
    naive_->ComputeFockMatrix();
    
    CompareMatrices(multi_mod_noprune);
    
    
    Destruct();
    
    
  
  } // TestMatricesMidNoPrune
  
  void TestMatricesMidPrune() {
  
    struct datanode* multi_mod_prune = fx_submodule(NULL, "multi_prune", 
                                                      "multi_prune");
    
    fx_set_param(multi_mod_prune, "epsilon", "1.0");
    fx_set_param(multi_mod_prune, "leaf_size", "1");
    
    
    struct datanode* naive_mod_prune = fx_submodule(NULL, "naive_prune", 
                                                      "naive_prune");
    
    fx_set_param(naive_mod_prune, "coulomb_output", "naive_coulomb_10_1.0.csv");
    fx_set_param(naive_mod_prune, "exchange_output", 
                 "naive_exchange_10_1.0.csv");
    
    Setup("test_centers_10.csv", "test_density_10.csv", 
          multi_mod_prune, naive_mod_prune, 1.0);
    
    multi_->ComputeFockMatrix();
    Matrix cou;
    Matrix exc;
    multi_->OutputFockMatrix(NULL, &cou, &exc, &old_from_new);
    
    naive_->ComputeFockMatrix();
    
    CompareMatrices(multi_mod_prune);
    
    
    Destruct();    
  
  } // TestMatricesMidPrune
  
  void TestMatricesLarge() {
  
    struct datanode* multi_mod_large = fx_submodule(NULL, "multi_large", 
                                                    "multi_large");
    
    fx_set_param(multi_mod_large, "epsilon", "0.05");
    fx_set_param(multi_mod_large, "leaf_size", "5");
    fx_set_param(multi_mod_large, "hybrid_cutoff", "50");
    fx_set_param(multi_mod_large, "epsilon_absolute", "25");
    
    
    struct datanode* naive_mod_large = fx_submodule(NULL, "naive_large", 
                                                    "naive_large");
    
    fx_set_param(naive_mod_large, "coulomb_output", 
                 "naive_coulomb_100_0.01.csv");
    fx_set_param(naive_mod_large, "exchange_output", 
                 "naive_exchange_100_0.01.csv");
    
    Setup("test_centers_100.csv", "test_density_100.csv", 
          multi_mod_large, naive_mod_large, 0.01);
    
    fx_timer_start(multi_mod_large, "multi");
    multi_->ComputeFockMatrix();
    fx_timer_stop(multi_mod_large, "multi");
    Matrix cou;
    Matrix exc;
    multi_->OutputFockMatrix(NULL, &cou, &exc, &old_from_new);
    
    fx_timer_start(naive_mod_large, "naive");
    naive_->ComputeFockMatrix();
    fx_timer_stop(naive_mod_large, "naive");
    
    CompareMatrices(multi_mod_large);
    
    
    Destruct();    
  
  } // TestMatricesLarge
  
  void TestMatrices() {
  
    NONFATAL("SMALL Test\n");
    TestMatricesSmall();
    
    NONFATAL("NOPRUNE Test\n");
    TestMatricesMidNoPrune();
    
    NONFATAL("PRUNE Test\n");
    TestMatricesMidPrune();
    
    
    if (fx_param_exists(NULL, "large")) {
      NONFATAL("LARGE Test\n");
      TestMatricesLarge();
    }
  } // TestMatrices
  
private:
  
  DualTreeIntegrals* multi_;
  
  NaiveFockMatrix* naive_;
  
  ArrayList<index_t> old_from_new;
  
}; //class FockMatrixTest


int main(int argc, char* argv[]) {
  
  fx_init(argc, argv);
  
  FockMatrixTest tester;
  tester.TestMatrices();
  
  fx_done();
  
  return 0;
  
} // main