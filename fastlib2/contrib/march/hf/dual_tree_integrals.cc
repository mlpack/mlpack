#include "dual_tree_integrals.h"
#include "naive_fock_matrix.h"

int main(int argc, char* argv[]) {


  fx_init(argc, argv);
  
  DualTreeIntegrals integrals;
  
  const char* centers_file = fx_param_str(NULL, "centers", "test_centers.csv");
  Matrix centers_in;
  data::Load(centers_file, &centers_in);
  
  
  /*for (int i = 0; i < centers_in.n_cols(); i++) {
    
    Vector test_vec1; 
    centers_in.MakeColumnVector(i, &test_vec1);
    
    for (int j = i; j < centers_in.n_cols(); j++) {
    
      Vector test_vec2;
      centers_in.MakeColumnVector(j, &test_vec2);
      
      double dist = la::DistanceSqEuclidean(test_vec1, test_vec2);
      printf("dist(%d, %d) = %g\n", i, j, dist);
      
    }
    
  }
  */
  
  index_t num_funs = centers_in.n_cols();
  
  la::Scale((double)num_funs, &centers_in);
  
  data::Save("gaussian_test300.csv", centers_in);
  
  Matrix density_in;
  //data::Load("test_density.csv", &density_in);
  density_in.Init(num_funs, num_funs);
  density_in.SetAll(1.0);
  
  Matrix core_in;
  core_in.Init(num_funs, num_funs);
  core_in.SetZero();
  //data::Load("core_test.csv", &core_in);
  
  
  
  integrals.Init(centers_in, NULL, density_in, core_in);
  
  fx_timer_start(NULL, "multi_tree");
  integrals.ComputeFockMatrix();
  fx_timer_stop(NULL, "multi_tree");
  
  printf("MULTI-TREE:\n");
  integrals.OutputFockMatrix();
  
 /* NaiveFockMatrix naive;
  
  naive.Init(centers_in, NULL, density_in);
  
  fx_timer_start(NULL, "naive");
  naive.ComputeFockMatrix();
  fx_timer_stop(NULL, "naive");
  
  printf("\n\nNAIVE:\n");
  naive.PrintFockMatrix();
  
  printf("\n\n");
  */
  fx_done();

  return 0;
}