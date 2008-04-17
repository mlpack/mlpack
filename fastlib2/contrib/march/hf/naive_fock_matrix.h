/**
 * @file naive_fock_matrix.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Naive implementation of dynamic integral computation code.
 */

#ifndef NAIVE_FOCK_MATRIX_H
#define NAIVE_FOCK_MATRIX_H

#include <fastlib/fastlib.h>

/**
 * Algorithm class for naively computing the Fock matrix.  
 */
class NaiveFockMatrix {

 public:

  FORBID_ACCIDENTAL_COPIES(NaiveFockMatrix);

 public:
 
  NaiveFockMatrix() {}
  
  ~NaiveFockMatrix() {}

  
 private:
  
  // The repulsion part of the fock matrix
  Matrix repulsion_matrix_;
  
  // -1/2 * the exchange part of the matrix
  Matrix exchange_matrix_;
  
  // The total fock matrix = repulsion + exchange
  Matrix fock_matrix_;
  
  // The centers of the Gaussian basis functions
  Matrix centers_;
  
  // The density matrix
  Matrix densities_;
  
  // The global bandwidth of all the spherical Gaussians
  double bandwidth_;
  
  // The fx module
  struct datanode* module_;
  
  // The total number of basis functions
  int number_of_basis_functions_;
  
  // The single, global normalization constant for all basis functions 
  double normalization_constant_fourth_;
  
  /**
   * The ERF-like function from my notes
   */
  double F_(double z) {
  
    if (z == 0) {
      return 1.0;
    }
    else {
      return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
    }
  
  } // F_()
  
  double SingleIntegral_(int i, int j, int k, int l) {
  
    Vector i_vec;
    Vector j_vec;
    Vector k_vec;
    Vector l_vec;
    
    centers_.MakeColumnVector(i, &i_vec);
    centers_.MakeColumnVector(j, &j_vec);
    centers_.MakeColumnVector(k, &k_vec);
    centers_.MakeColumnVector(l, &l_vec);
    
    double ik_dist;
    double jl_dist;
    
    double ij_dist;
    double kl_dist;
    
    
    ik_dist = la::DistanceSqEuclidean(i_vec, k_vec);
    jl_dist = la::DistanceSqEuclidean(j_vec, l_vec);
    
    ij_dist = la::DistanceSqEuclidean(i_vec, j_vec);
    kl_dist = la::DistanceSqEuclidean(k_vec, l_vec);
    
    double integral_value = normalization_constant_fourth_;
    
    integral_value = integral_value * 0.25 * pow((math::PI/bandwidth_), 2.5);
    
    integral_value = integral_value * 
        F_(bandwidth_ * 0.25 * (ik_dist + jl_dist));
        
    integral_value = integral_value * 
        exp(-0.5 * bandwidth_ * (ij_dist + kl_dist));
    
    return integral_value; 
  
  } // SingleIntegral_()
  
  
  
 public:

  /**
   * Initialize the class 
   */
  void Init(const Matrix& centers_in, struct datanode* mod, 
            const Matrix& density_in) {
    
    centers_.Copy(centers_in);
    
    densities_.Copy(density_in);
    
    module_ = mod;
    
    bandwidth_ = fx_param_double(module_, "bandwidth", 0.1);
    
    number_of_basis_functions_ = centers_.n_cols();
    
    repulsion_matrix_.Init(number_of_basis_functions_, 
                           number_of_basis_functions_);
    
    repulsion_matrix_.SetAll(0.0);
    
    exchange_matrix_.Init(number_of_basis_functions_, 
                          number_of_basis_functions_);
    
    exchange_matrix_.SetAll(0.0);
    
    normalization_constant_fourth_ = pow((2 * bandwidth_ / math::PI), 3);
    
  } // Init()
  

  /**
   * Computes the Fock Matrix
   */
  void ComputeFockMatrix() {
  
    double largest_integral = 0.0;
    double smallest_integral = DBL_MAX;
  
    for (int i = 0; i < number_of_basis_functions_; i++) {
    
      for (int j = i; j < number_of_basis_functions_; j++) {
      
        double current_integral = 0.0;
      
        for (int k = 0; k < number_of_basis_functions_; k++) {
        
          for (int l = k; l < number_of_basis_functions_; l++) {
          
            double one_integral = SingleIntegral_(i, j, k, l);
          
            if (one_integral > largest_integral) {
              largest_integral = one_integral;
            }
            if (one_integral < smallest_integral) {
              smallest_integral = one_integral;
            }
            
            if (k != l) {
              one_integral = 2 * one_integral;
            }
            
            
            if (j != i) {
              current_integral = current_integral + 
                  2 * densities_.ref(i, j) * one_integral;
            }
            else {
              current_integral = current_integral + 
                 densities_.ref(i, j) * one_integral;
            }
          } // l
        
        } // k
        
        repulsion_matrix_.set(i, j, current_integral);
        repulsion_matrix_.set(j, i, current_integral);
      
      } // j
    
    } // i
    
    la::AddInit(repulsion_matrix_, exchange_matrix_, &fock_matrix_);
    
  //  printf("smallest_integral = %g, largest_integral = %g\n", smallest_integral, largest_integral);
    
  } // ComputeFockMatrix()
  
  /**
   * Output the Fock Matrix for comparison to my algorithm
   */
  void PrintFockMatrix() {
  
    double average_value = 0.0;
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
     
      for (index_t j = 0; j < number_of_basis_functions_; j++) {
      
        average_value = average_value + fock_matrix_.ref(i, j);
      
      }
    }
    
    average_value = average_value/(number_of_basis_functions_ * number_of_basis_functions_);
    
    fx_format_result(module_, "average_matrix_value", "%g", average_value);
  
    //fock_matrix_.PrintDebug();
  
  } // PrintFockMatrix()

}; // NaiveFockMatrix

#endif