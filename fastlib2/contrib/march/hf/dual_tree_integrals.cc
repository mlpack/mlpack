#include "dual_tree_integrals.h"
#include "naive_fock_matrix.h"



/**
* Computes the function F from my notes, which is similar to erf 
 *
 * TODO: Consider inlining this function
 *
 * Also, the integral project notes have a slightly different definition of
 * this function.  I should make sure they're compatible.  
 */
double DualTreeIntegrals::ErfLikeFunction_(double z) {
  
  if (z == 0) {
    return 1.0;
  }
  else {
    return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
  }
  
} // ErfLikeFunction_


/**
* Computes a single integral based on the distances between the centers.  
 * Intended for use in bounding.  
 */
double DualTreeIntegrals::ComputeSingleIntegral_(double mu_nu_dist, 
                                                 double rho_sigma_dist, 
                                                 double four_way_dist) {
  
  //printf("Called pruning integral\n");
  
  double return_value;
  
  return_value = 0.25 * pow((math::PI/bandwidth_), 2.5) * 
    normalization_constant_fourth_;
  
  // the 0.25 comes from the four center distance identity
  return_value = return_value * ErfLikeFunction_(bandwidth_ * four_way_dist);
  
  // I added a factor of 1/2, another mistake I think
  return_value = return_value * 
    exp(-0.5 * bandwidth_ * (mu_nu_dist + rho_sigma_dist));
  
  
  
  return return_value;
  
  
} // ComputeSingleIntegral_

/**
* Finds the single integral of four Gaussians at the given centers
 *
 *
 * Am I including the density matrix in these computations?
 *
 * TODO: consider rewriting this to take the distances as the arguments so 
 * that it can be used for pruning computations as well.  
 *
 * Also, rewrite the math to take advantage of the mixed centers identity
 */
double DualTreeIntegrals::ComputeSingleIntegral_(const Vector& mu_center, 
                                                 const Vector& nu_center, 
                                                 const Vector& rho_center, 
                                                 const Vector& sigma_center) {
  
  // Should just be able to plug in the formula
  
  // Constant in front
  double return_value = 0.25 * pow((math::PI/bandwidth_), 2.5) * 
  normalization_constant_fourth_;
  
  //double four_centers_dists = la::DistanceSqEuclidean(mu_center, rho_center) + 
  //la::DistanceSqEuclidean(nu_center, sigma_center);
  
  Vector mu_plus_nu;
  la::AddInit(mu_center, nu_center, &mu_plus_nu);
  la::Scale(0.5, &mu_plus_nu);
  Vector rho_plus_sigma;
  la::AddInit(rho_center, sigma_center, &rho_plus_sigma);
  la::Scale(0.5, &rho_plus_sigma);
  double four_centers_dists = 
    la::DistanceSqEuclidean(mu_plus_nu, rho_plus_sigma);
  
  // F(\alpha d^2(x_{i j} x_{k l}))
  // equivalent to F(\alpha 
  return_value = return_value * 
    ErfLikeFunction_(bandwidth_ * four_centers_dists);
  
  
  // exp(-\alpha (d^2(x_i, x_j) + d^2(x_k, x_l))
  double between_centers_dists = 
    la::DistanceSqEuclidean(mu_center, nu_center) 
    + la::DistanceSqEuclidean(rho_center, sigma_center);
  
  return_value = return_value * 
    exp(-0.5 * bandwidth_ * between_centers_dists);
  
  //printf("computing integral: %g\n", return_value);
  
  // printf("between centers dist: %g, integral: %g\n", between_centers_dists, return_value);
  
  
  return return_value;
  
  } // ComputeSingleIntegral_
  
  
  
bool DualTreeIntegrals::RectangleOnDiagonal_(IntegralTree* mu, 
                                             IntegralTree* nu) {

  if (mu == nu) {
    return false;
  }
  else if (mu->begin() >= nu->end()){
    return false;
  }
  else {
    return true;
  }

} // RectangleOnDiagonal_()


index_t DualTreeIntegrals::CountOnDiagonal_(SquareIntegralTree* rho_sigma) {
                                            
  index_t on_diagonal;
  
  // one of these should be square, which one?
  SquareIntegralTree* left_child = rho_sigma->left();
  SquareIntegralTree* right_child = rho_sigma->right();
  
  if (left_child->query1() == left_child->query2()) {
    on_diagonal = left_child->query1()->count() * left_child->query2()->count();
  }
  else {
    DEBUG_ASSERT(right_child->query1() == right_child->query2());
    on_diagonal = right_child->query1()->count() * 
                  right_child->query2()->count();
  }
     
  return on_diagonal;                                   

} // CountOnDiagonal_()


/*
int main(int argc, char* argv[]) {


  fx_init(argc, argv, NULL);
  
  DualTreeIntegrals integrals;
  
  const char* centers_file = fx_param_str(NULL, "centers", "test_centers.csv");
  Matrix centers_in;
  data::Load(centers_file, &centers_in);
  
  index_t num_funs = centers_in.n_cols();
  
  la::Scale((double)num_funs, &centers_in);
  
  data::Save("gaussian_test300.csv", centers_in);
  
  const char* density_file = fx_param_str(NULL, "density", "test_density.csv");
  Matrix density_in;
  data::Load(density_file, &density_in);
  
  
  Matrix core_in;
  core_in.Init(num_funs, num_funs);
  core_in.SetZero();
  
  double bandwidth = fx_param_double(NULL, "bandwidth", 100.0);
  
  struct datanode* dual_mod = fx_submodule(NULL, "multi", "multi_tree");
  
  integrals.Init(centers_in, dual_mod, density_in, core_in, bandwidth);
  
  fx_timer_start(dual_mod, "multi_tree");
  integrals.ComputeFockMatrix();
  fx_timer_stop(dual_mod, "multi_tree");
  
  Matrix tree_coulomb;
  Matrix tree_exchange;
  ArrayList<index_t> old_from_new;
  
  printf("MULTI-TREE:\n");
  integrals.OutputFockMatrix(&tree_coulomb, &tree_exchange, &old_from_new);
  
  Matrix naive_coulomb;
  Matrix naive_exchange;
  if (data::Load("naive_exchange.csv", &naive_exchange) == SUCCESS_FAIL) {
    // printf("failed to load");
    naive_exchange.Destruct();
    NaiveFockMatrix naive;
  
    struct datanode* naive_mod = fx_submodule(NULL, "naive", "naive");
    naive.Init(centers_in, naive_mod, density_in, core_in, bandwidth);
  
    fx_timer_start(naive_mod, "naive");
    naive.ComputeFockMatrix();
    fx_timer_stop(naive_mod, "naive");
  
    
  
    printf("\n\nNAIVE:\n");
    naive.PrintFockMatrix(&naive_coulomb, &naive_exchange);
    
    data::Save("naive_coulomb.csv", naive_coulomb);
    data::Save("naive_exchange.csv", naive_exchange);
  
    printf("\n\n");
  }
  else {
    data::Load("naive_coulomb.csv", &naive_coulomb);
  }
  

  //printf("\n\n DIFFERENCE:\n");
  Matrix difference_mat;
  
  la::Scale(-1.0, &naive_coulomb);
  difference_mat.Copy(tree_coulomb);
  
  Vector diff;
  for (index_t i = 0; i < difference_mat.n_cols(); i++) {
  
  
  
  } // i
  
  double coulomb_error = 0.0;
  double max_coulomb_error = 0.0;
  double exchange_error = 0.0;
  double max_exchange_error = 0.0;
  
  for (index_t i = 0; i < tree_exchange.n_cols(); i++) {
  
    for (index_t j = 0; j < tree_exchange.n_rows(); j++) {
    
      double this_val = fabs(naive_exchange.get(old_from_new[i], old_from_new[j]) - 
          tree_exchange.get(i, j));
      //printf("i:%d, j:%d, this_val:%g\n", i, j, this_val);
      exchange_error = exchange_error + this_val;
      
      if (this_val > max_exchange_error) { 
        max_exchange_error = this_val;
        //printf("max_error: (%d, %d)\n", i, j);
      } 
      
      this_val = fabs(naive_coulomb.get(old_from_new[i], old_from_new[j]) - 
                             tree_coulomb.get(i, j));
      //printf("i:%d, j:%d, this_val:%g\n", i, j, this_val);
      coulomb_error = coulomb_error + this_val;
      
      if (this_val > max_coulomb_error) { 
        max_coulomb_error = this_val;
        //printf("max_error: (%d, %d)\n", i, j);
      }     
                  
    } // j
  
  } // i 
  
  exchange_error = exchange_error/(tree_exchange.n_cols() * tree_exchange.n_rows());
  coulomb_error = coulomb_error/(tree_exchange.n_cols() * tree_exchange.n_rows());
  
  fx_format_result(NULL, "ave_abs_coulomb_error", "%g", coulomb_error);
  fx_format_result(NULL, "max_abs_coulomb_error", "%g", max_coulomb_error);
    
  fx_format_result(NULL, "ave_abs_exchange_error", "%g", exchange_error);
  fx_format_result(NULL, "max_abs_exchange_error", "%g", max_exchange_error);

  fx_done(NULL);

  return 0;

}
*/