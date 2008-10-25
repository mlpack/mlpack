#ifndef MULTIBODY_KERNEL_CARTESIAN_H
#define MULTIBODY_KERNEL_CARTESIAN_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "mlpack/series_expansion/inverse_pow_dist_kernel_aux.h"

#include "../multitree_template/upper_triangular_square_matrix.h"

/** @brief This class represents a 3-body interaction.
 */
class MultiInversePowDistKernelAux {

 public:

  double coeff;
  
  InversePowDistGradientKernelAux inverse_pow_dist_gradient_kernel_aux;

  InversePowDistKernelAux first_inverse_pow_dist_kernel_aux;
  
  InversePowDistKernelAux second_inverse_pow_dist_kernel_aux;

  OT_DEF_BASIC(MultiInversePowDistKernelAux) {
    OT_MY_OBJECT(coeff);
    OT_MY_OBJECT(inverse_pow_dist_gradient_kernel_aux);
    OT_MY_OBJECT(first_inverse_pow_dist_kernel_aux);
    OT_MY_OBJECT(second_inverse_pow_dist_kernel_aux);
  }

 public:
  void Init(double coeff_in, int first_order, int second_order, 
	    int third_order, int series_truncation_order, int dim) {

    coeff = coeff_in;
    inverse_pow_dist_gradient_kernel_aux.Init
      (first_order, series_truncation_order, dim);
    first_inverse_pow_dist_kernel_aux.Init
      (second_order, series_truncation_order, dim);
    second_inverse_pow_dist_kernel_aux.Init
      (third_order, series_truncation_order, dim);
    
  }

  double Evaluate(int dimension, const double *point1, const double *point2,
		  double squared_distance1, double squared_distance2,
		  double squared_distance3) const {
    
    return coeff * inverse_pow_dist_gradient_kernel_aux.kernel_.
      EvalUnnorm(point1, point2, squared_distance1) *
      first_inverse_pow_dist_kernel_aux.kernel_.EvalUnnormOnSq
      (squared_distance2) *
      second_inverse_pow_dist_kernel_aux.kernel_.EvalUnnormOnSq
      (squared_distance3);
  }

  static inline double Evaluate
  (double coeff_in, int first_order, int second_order, 
   int third_order, int dimension, const double *point1, const double *point2,
   double squared_distance1, double squared_distance2,
   double squared_distance3) {
    
    return coeff_in * 
      InversePowDistGradientKernel::EvalUnnorm(dimension, first_order,
					       point1, point2, 
					       squared_distance1) *
      InversePowDistKernel::EvalUnnormOnSq(second_order, squared_distance2) *
      InversePowDistKernel::EvalUnnormOnSq(third_order, squared_distance3);
  }
};

class AxilrodTellerForceKernelAux {

 private:

  double AxilrodTellerForceKernelPositiveEvaluate
  (int dimension_in, const double *first_point, const double *second_point,
   double first_distance, double first_distance_pow_three,
   double first_distance_pow_five, double first_distance_pow_seven,
   double second_distance, double second_distance_pow_three,
   double second_distance_pow_five, double third_distance,
   double third_distance_pow_three, double third_distance_pow_five) {
    
    double first_factor = second_distance / third_distance_pow_five *
      (first_point[dimension_in] - second_point[dimension_in]);
    double second_factor = third_distance / second_distance_pow_five *
      (first_point[dimension_in] - second_point[dimension_in]);
    return 1.875 * (first_factor + second_factor) /
      first_distance_pow_seven;
  }

  double AxilrodTellerForceKernelNegativeEvaluate
  (int dimension_in,
   const double *first_point,
   const double *second_point,
   double first_distance,
   double first_distance_pow_three,
   double first_distance_pow_five,
   double first_distance_pow_seven,
   double second_distance,
   double second_distance_pow_three,
   double second_distance_pow_five,
   double third_distance,
   double third_distance_pow_three,
   double third_distance_pow_five) {
    
    double coord_diff = first_point[dimension_in] - 
      second_point[dimension_in];
    
    double first_factor = (-0.75) / 
      (first_distance_pow_five * second_distance_pow_three *
       third_distance_pow_three);
    double second_factor = (-0.375) /
      (first_distance * second_distance_pow_five * third_distance_pow_five);
    double third_factor = (-0.375) / 
      (first_distance_pow_three * second_distance_pow_three *
       third_distance_pow_five);
    double fourth_factor = (-0.375) /
      (first_distance_pow_three * second_distance_pow_five *
       third_distance_pow_three);
    double fifth_factor = (-1.125) /
      (first_distance_pow_five * second_distance * third_distance_pow_five);
    double sixth_factor = (-1.125) /
      (first_distance_pow_five * second_distance_pow_five *
       third_distance);
    double seventh_factor = (-1.875) /
      (first_distance_pow_seven * second_distance *
       third_distance_pow_three);
    double eigth_factor = (-1.875) /
      (first_distance_pow_seven * second_distance_pow_three *
       third_distance);
    
    // The eight negative components.
    return coord_diff * 
      (first_factor + second_factor + third_factor + fourth_factor +
       fifth_factor + sixth_factor + seventh_factor + eigth_factor);
  }

  static const int series_truncation_order_ = 0;

  static const int dimension_ = 3;
  
  static const int order_ = 3;

  Matrix lower_bound_squared_distances_;
  
  Matrix upper_bound_squared_distances_;

  Matrix positive_contributions_;

  Matrix negative_contributions_;

  GenVector<int> first_index;
  GenVector<int> second_index;
  GenVector<int> third_index;

 public:
  
  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  AxilrodTellerForceKernelAux() {   
  }

  /** @brief The default destructor.
   */
  ~AxilrodTellerForceKernelAux() {
  }

  ////////// User-level Functions //////////

  /** @brief The initializer of the kernel.
   */
  void Init() {
    
    lower_bound_squared_distances_.Init(order_, order_);
    upper_bound_squared_distances_.Init(order_, order_);

    // Allocate the space for storing temporary computation results.
    positive_contributions_.Init(3, 3);    
    negative_contributions_.Init(3, 3);

    first_index.Init(3);
    second_index.Init(3);
    third_index.Init(3);

    // Hard-code the index permutation...
    first_index[0] = 0;
    first_index[1] = 0;
    first_index[2] = 2;
    second_index[0] = 1;
    second_index[1] = 2;
    second_index[2] = 1;
    third_index[0] = 2;
    third_index[1] = 1;
    third_index[2] = 0;
  }
  
  void EvaluateHelper
  (const ArrayList<Matrix *> &sets,
   const ArrayList<index_t> &indices,
   const Matrix &squared_distances) {

    for(index_t p = 0; p < order_; p++) {

      double *positive_contribution =
	positive_contributions_.GetColumnPtr(p);
      double *negative_contribution = 
	negative_contributions_.GetColumnPtr(p);
      
      const double *first_point = sets[first_index[p]]->GetColumnPtr
	(indices[first_index[p]]);
      const double *second_point = sets[second_index[p]]->GetColumnPtr
	(indices[second_index[p]]);
      
      double distance_first_second =
	math::Pow<1, 2>(squared_distances.get(first_index[p],
					      second_index[p]));
      double distance_first_second_pow_three =
	distance_first_second *
	squared_distances.get(first_index[p], second_index[p]);
      double distance_first_second_pow_five =
	squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_three;
      double distance_first_second_pow_seven =
	squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_five;
      double distance_first_third =
	math::Pow<1, 2>(squared_distances.get(first_index[p],
					      third_index[p]));
      double distance_first_third_pow_three =
	distance_first_third *
	squared_distances.get(first_index[p], third_index[p]);
      double distance_first_third_pow_five =
	squared_distances.get(first_index[p], third_index[p]) *
	distance_first_third_pow_three;
      double distance_second_third =
	math::Pow<1, 2>(squared_distances.get(second_index[p],
					      third_index[p]));
      double distance_second_third_pow_three =
	distance_second_third *
	squared_distances.get(second_index[p], third_index[p]);
      double distance_second_third_pow_five =
	squared_distances.get(second_index[p], third_index[p]) *
	distance_second_third_pow_three;

      for(index_t counter = 0; counter < dimension_; counter++) {
	positive_contribution[counter] += 
	  AxilrodTellerForceKernelPositiveEvaluate
	  (counter, first_point, second_point,
	   distance_first_second,
	   distance_first_second_pow_three,
	   distance_first_second_pow_five,
	   distance_first_second_pow_seven,
	   distance_first_third,
	   distance_first_third_pow_three,
	   distance_first_third_pow_five,
	   distance_second_third,
	   distance_second_third_pow_three,
	   distance_second_third_pow_five);

	negative_contribution[counter] += 
	  AxilrodTellerForceKernelNegativeEvaluate
	  (counter, first_point, second_point,
	   distance_first_second,
	   distance_first_second_pow_three,
	   distance_first_second_pow_five,
	   distance_first_second_pow_seven,
	   distance_first_third,
	   distance_first_third_pow_three,
	   distance_first_third_pow_five,
	   distance_second_third,
	   distance_second_third_pow_three,
	   distance_second_third_pow_five);
      }
    }
  };

  template<typename Global>
  void Evaluate(const ArrayList<Matrix *> &sets,
		const Matrix &squared_distances, Global &globals) {

    // Clear the contribution accumulator.
    positive_contributions_.SetZero();
    negative_contributions_.SetZero();

    // Evaluate the positive kernels at (0, 1), (0, 2), (1, 2) and (0,
    // 2), (0, 1), (1, 2) and (1, 2), (0, 2), (0, 1) in dimension 0,
    // 1, 2.
    EvaluateHelper(sets, globals.chosen_indices, squared_distances);
  }

  void PairwiseEvaluateLoop(const ArrayList<Matrix *> &sets,
			    const ArrayList<index_t> &indices,
			    Matrix &squared_distances) {
    
    // Evaluate the squared distance between (first_index, start).
    for(index_t first_index = 0; first_index < order_ - 1; first_index++) {
      const Matrix *first_set = sets[first_index];
      
      for(index_t second_index = first_index + 1; second_index < order_;
	  second_index++) {
	
	const Matrix *second_set = sets[second_index];
	double squared_distance =
	  la::DistanceSqEuclidean
	  (first_set->n_rows(), 
	   first_set->GetColumnPtr(indices[first_index]),
	   second_set->GetColumnPtr(indices[second_index]));
	squared_distances.set(first_index, second_index, squared_distance);
      }
    }
  }

  /** @brief Evaluate the kernel given the chosen indices exhaustively.
   */
  template<typename Global>
  void Evaluate(const ArrayList<Matrix *> &sets, Global &globals) {
    
    // Evaluate the pairwise distances.    
    PairwiseEvaluateLoop(sets, globals.chosen_indices,
			 lower_bound_squared_distances_);
    
    // Add the contribution...
    Evaluate(sets, lower_bound_squared_distances_, globals);
    
  }
  
};

#endif
