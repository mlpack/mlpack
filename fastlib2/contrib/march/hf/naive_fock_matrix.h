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
  
  friend class FockMatrixTest;

 public:
 
  NaiveFockMatrix() {}
  
  ~NaiveFockMatrix() {}

  
 private:
  
  // The Coulomb part of the fock matrix
  Matrix coulomb_matrix_;
  
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
    
    
    double ij_dist;
    double kl_dist;
    
    Vector i_plus_j;
    la::AddInit(i_vec, j_vec, &i_plus_j);
    la::Scale(0.5, &i_plus_j);
    Vector k_plus_l;
    la::AddInit(k_vec, l_vec, &k_plus_l);
    la::Scale(0.5, &k_plus_l);
    
    double four_way_dist = la::DistanceSqEuclidean(i_plus_j, k_plus_l);
        
    ij_dist = la::DistanceSqEuclidean(i_vec, j_vec);
    kl_dist = la::DistanceSqEuclidean(k_vec, l_vec);
    //printf("%g, %g, %g, %g\n", ik_dist, jl_dist, ij_dist, kl_dist);
    
    double integral_value = normalization_constant_fourth_;
    
    integral_value = integral_value * 0.25 * pow((math::PI/bandwidth_), 2.5);
    
    integral_value = integral_value * 
        F_(bandwidth_ * four_way_dist);
        
    integral_value = integral_value * 
        exp(-0.5 * bandwidth_ * (ij_dist + kl_dist));
    //printf("%d, %d, %d, %d, %g\n", i, j, k, l, integral_value);
    
    /*if (i == 0 && j == 2 && k == 1 && l == 2) {
      printf("val: %g\n", integral_value);
    }
    */
    return integral_value; 
  
  } // SingleIntegral_()
  
  
  
 public:

  /**
   * Initialize the class 
   */
  void Init(const Matrix& centers_in, struct datanode* mod, 
            const Matrix& density_in, double band) {
    
    centers_.Copy(centers_in);
    
    densities_.Copy(density_in);
    
    module_ = mod;
    
    bandwidth_ = band;
    
    number_of_basis_functions_ = centers_.n_cols();
    
    coulomb_matrix_.Init(number_of_basis_functions_, 
                           number_of_basis_functions_);
    
    coulomb_matrix_.SetAll(0.0);
    
    exchange_matrix_.Init(number_of_basis_functions_, 
                          number_of_basis_functions_);
    
    exchange_matrix_.SetAll(0.0);
    
    fock_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    
    fock_matrix_.SetAll(0.0);
    
    normalization_constant_fourth_ = pow((2 * bandwidth_ / math::PI), 3);
    fx_format_result(module_, "normalization", "%g", 
                     normalization_constant_fourth_);
    fx_format_result(module_, "bandwidth", "%g", bandwidth_);
    
    
  } // Init()
  
  void UpdateMatrices(const Matrix& new_density) {
    
    densities_.CopyValues(new_density);
    
    coulomb_matrix_.Destruct();
    exchange_matrix_.Destruct();
    fock_matrix_.Destruct();
    
  } 
  

  /**
   * Computes the Fock Matrix
   */
  void ComputeFockMatrix() {
  
    
    const char* coulomb_file = fx_param_str(module_, "coulomb_output", 
                                            "/dev/null/blah");
    const char* exchange_file = fx_param_str(module_, "exchange_output", 
                                             "/dev/null/blah");
    
    
    if (data::Load(coulomb_file, &coulomb_matrix_) == SUCCESS_FAIL) {
    
      coulomb_matrix_.Destruct();
      coulomb_matrix_.Init(number_of_basis_functions_, 
                           number_of_basis_functions_);
      coulomb_matrix_.SetZero();
      
      exchange_matrix_.Init(number_of_basis_functions_, 
                            number_of_basis_functions_);
      
      exchange_matrix_.SetZero();

      /*
      double largest_integral = 0.0;
      double smallest_integral = DBL_MAX;
       */
    
      for (int i = 0; i < number_of_basis_functions_; i++) {
      
        for (int j = 0; j < number_of_basis_functions_; j++) {
        
          double current_integral = 0.0;
        
          for (int k = 0; k < number_of_basis_functions_; k++) {
          
            for (int l = 0; l < number_of_basis_functions_; l++) {
            
              double one_integral = SingleIntegral_(i, j, k, l);

              /*
              if (one_integral > largest_integral) {
                largest_integral = one_integral;
              }
              if (one_integral < smallest_integral) {
                smallest_integral = one_integral;
              }
              */
              current_integral = current_integral + 
                                 (densities_.ref(k, l) * one_integral);
                                 
                                 
              double exchange_integral = exchange_matrix_.ref(i, j);
              exchange_integral = exchange_integral + 
                  (SingleIntegral_(i, k, j, l) * densities_.ref(k, l));
                  
              exchange_matrix_.set(i, j, exchange_integral);
                
            } // l
          
          } // k
          
          coulomb_matrix_.set(i, j, current_integral);
        
        } // j
      
      } // i
      
      la::Scale(0.5, &exchange_matrix_);
      /*
      data::Save(coulomb_file, coulomb_matrix_);
                                               
      data::Save(exchange_file, exchange_matrix_);
      */
    }
    else {
    
      data::Load(exchange_file, &exchange_matrix_);
      
    }
    
    la::SubInit(exchange_matrix_, coulomb_matrix_, &fock_matrix_);

  } // ComputeFockMatrix()
  
  /**
   * Output the Fock Matrix for comparison to my algorithm
   */
  void PrintFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                       Matrix* exchange_out) {
  /*
    double average_value = 0.0;
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
     
      for (index_t j = 0; j < number_of_basis_functions_; j++) {
      
        average_value = average_value + fock_matrix_.ref(i, j);
      
      }
    }
    
    average_value = 
        average_value/(number_of_basis_functions_ * number_of_basis_functions_);
    
    fx_format_result(module_, "average_matrix_value", "%g", average_value);
  */
    /*printf("Coulomb (naive):\n");
    coulomb_matrix_.PrintDebug();
    
    printf("Naive Exchange:\n");
    exchange_matrix_.PrintDebug();
    */
    if (fock_out) {
      fock_out->Copy(fock_matrix_);
    }
    if (coulomb_out) {
      coulomb_out->Copy(coulomb_matrix_);
    }
    if (exchange_out) {
      exchange_out->Copy(exchange_matrix_);
    }
    
  
  } // PrintFockMatrix()

}; // NaiveFockMatrix

#endif