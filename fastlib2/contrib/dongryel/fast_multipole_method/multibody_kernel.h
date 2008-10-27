#ifndef MULTIBODY_KERNEL_CARTESIAN_H
#define MULTIBODY_KERNEL_CARTESIAN_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "mlpack/series_expansion/inverse_pow_dist_kernel_aux.h"

#include "../multitree_template/upper_triangular_square_matrix.h"

class AxilrodTellerForceKernelAux {

 private:

  // This only works for kd-tree, but perhaps could be generalized to
  // ball-type bounds...
  template<typename TBound>
  void DifferenceBetweenTwoRanges(int dimension_in,
				  const TBound &first_bound,
				  const TBound &second_bound,
				  DRange &result) {
    
    const DRange &first_range = first_bound.get(dimension_in);
    const DRange &second_range = second_bound.get(dimension_in);
    
    double diff1 = first_range.hi - second_range.lo;
    double diff2 = first_range.lo - second_range.hi;
    result.lo = std::min(diff1, diff2);
    result.hi = std::max(diff1, diff2);
  }

  template<typename TBound>
  void AxilrodTellerForceKernelPositiveEvaluate
  (int dimension_in, const TBound &first_bound, const TBound &second_bound,
   const DRange &first_distance,
   const DRange &first_distance_pow_three,
   const DRange &first_distance_pow_five,
   const DRange &first_distance_pow_seven,
   const DRange &second_distance,
   const DRange &second_distance_pow_three,
   const DRange &second_distance_pow_five,
   const DRange &third_distance,
   const DRange &third_distance_pow_three,
   const DRange &third_distance_pow_five,
   DRange &negative_contribution, DRange &positive_contribution) {
    
    DRange first_factor, second_factor;
    first_factor.lo = 1.875 * second_distance.lo / 
      (third_distance_pow_five.hi * first_distance_pow_seven.hi);
    first_factor.hi = 1.875 * second_distance.hi /
      (third_distance_pow_five.lo * first_distance_pow_seven.lo);
    second_factor.lo = 1.875 * third_distance.lo / 
      (second_distance_pow_five.hi * first_distance_pow_seven.hi);
    second_factor.hi = 1.875 * third_distance.hi /
      (second_distance_pow_five.lo * first_distance_pow_seven.lo);

    DRange coord_range;
    DifferenceBetweenTwoRanges(dimension_in, first_bound, second_bound,
			       coord_range);
    DEBUG_ASSERT(coord_range.lo <= coord_range.hi);

    // If the minimum is positive,
    if(coord_range.lo >= 0) {
      negative_contribution.lo = negative_contribution.hi = 0;
      positive_contribution.lo = (first_factor.lo + second_factor.lo) *
	coord_range.lo;
      positive_contribution.hi = (first_factor.hi + second_factor.hi) *
	coord_range.hi;
    }

    // If the maximum is negative,
    else if(coord_range.hi <= 0) {
      negative_contribution.lo = (first_factor.hi + second_factor.hi) *
	coord_range.lo;
      negative_contribution.hi = (first_factor.lo + second_factor.lo) *
	coord_range.hi;
      positive_contribution.lo = positive_contribution.hi = 0;
    }

    // If the minimum is negative, maximum is positive,
    else {
      negative_contribution.lo = (first_factor.hi + second_factor.hi) *
	coord_range.lo;
      negative_contribution.hi = positive_contribution.lo = 0;
      positive_contribution.hi = (first_factor.hi + second_factor.hi) *
	coord_range.hi;
    }

    DEBUG_ASSERT(negative_contribution.hi <= 0 &&
		 negative_contribution.lo <= negative_contribution.hi);
    DEBUG_ASSERT(positive_contribution.lo >= 0 &&
		 positive_contribution.hi >= positive_contribution.lo);
  }

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

  template<typename TBound>
  void AxilrodTellerForceKernelNegativeEvaluate
  (int dimension_in, const TBound &first_bound, const TBound &second_bound,
   const DRange &first_distance, const DRange &first_distance_pow_three,
   const DRange &first_distance_pow_five, 
   const DRange &first_distance_pow_seven,
   const DRange &second_distance, const DRange &second_distance_pow_three,
   const DRange &second_distance_pow_five, const DRange &third_distance,
   const DRange &third_distance_pow_three, 
   const DRange &third_distance_pow_five, DRange &negative_contribution,
   DRange &positive_contribution) {

    // Compute the coordinate difference range.
    DRange coord_range;
    DifferenceBetweenTwoRanges(dimension_in, first_bound, second_bound,
			       coord_range);
    DEBUG_ASSERT(coord_range.lo <= coord_range.hi);

    DRange first_factor, second_factor, third_factor, fourth_factor,
      fifth_factor, sixth_factor, seventh_factor, eighth_factor;

    first_factor.lo = (-0.75) / 
      (first_distance_pow_five.lo * second_distance_pow_three.lo *
       third_distance_pow_three.lo);
    first_factor.hi = (-0.75) / 
      (first_distance_pow_five.hi * second_distance_pow_three.hi *
       third_distance_pow_three.hi);
    second_factor.lo = (-0.375) /
      (first_distance.lo * second_distance_pow_five.lo *
       third_distance_pow_five.lo);
    second_factor.hi = (-0.375) /
      (first_distance.hi * second_distance_pow_five.hi *
       third_distance_pow_five.hi);
    third_factor.lo = (-0.375) / 
      (first_distance_pow_three.lo * second_distance_pow_three.lo *
       third_distance_pow_five.lo);
    third_factor.hi = (-0.375) / 
      (first_distance_pow_three.hi * second_distance_pow_three.hi *
       third_distance_pow_five.hi);
    fourth_factor.lo = (-0.375) /
      (first_distance_pow_three.lo * second_distance_pow_five.lo *
       third_distance_pow_three.lo);
    fourth_factor.hi = (-0.375) /
      (first_distance_pow_three.hi * second_distance_pow_five.hi *
       third_distance_pow_three.hi);
    fifth_factor.lo = (-1.125) /
      (first_distance_pow_five.lo * second_distance.lo *
       third_distance_pow_five.lo);
    fifth_factor.hi = (-1.125) /
      (first_distance_pow_five.hi * second_distance.hi *
       third_distance_pow_five.hi);
    sixth_factor.lo = (-1.125) /
      (first_distance_pow_five.lo * second_distance_pow_five.lo *
       third_distance.lo);
    sixth_factor.hi = (-1.125) /
      (first_distance_pow_five.hi * second_distance_pow_five.hi *
       third_distance.hi);
    seventh_factor.lo = (-1.875) /
      (first_distance_pow_seven.lo * second_distance.lo *
       third_distance_pow_three.lo);
    seventh_factor.hi = (-1.875) /
      (first_distance_pow_seven.hi * second_distance.hi *
       third_distance_pow_three.hi);
    eighth_factor.lo = (-1.875) /
      (first_distance_pow_seven.lo * second_distance_pow_three.lo *
       third_distance.lo);
    eighth_factor.hi = (-1.875) /
      (first_distance_pow_seven.hi * second_distance_pow_three.hi *
       third_distance.hi);

    // If the minimum is positive,
    if(coord_range.lo >= 0) {
      positive_contribution.lo = positive_contribution.hi = 0;
      negative_contribution.lo = (first_factor.lo + second_factor.lo +
				  third_factor.lo + fourth_factor.lo +
				  fifth_factor.lo + sixth_factor.lo +
				  seventh_factor.lo + eighth_factor.lo) *
	coord_range.hi;
      negative_contribution.hi = (first_factor.hi + second_factor.hi +
				  third_factor.hi + fourth_factor.hi +
				  fifth_factor.hi + sixth_factor.hi +
				  seventh_factor.hi + eighth_factor.hi) *
	coord_range.lo;
    }

    // If the maximum is negative,
    else if(coord_range.hi <= 0) {
      positive_contribution.lo = (first_factor.hi + second_factor.hi +
				  third_factor.hi + fourth_factor.hi +
				  fifth_factor.hi + sixth_factor.hi +
				  seventh_factor.hi + eighth_factor.hi) *
	coord_range.hi;
      positive_contribution.hi = (first_factor.lo + second_factor.lo +
				  third_factor.lo + fourth_factor.lo +
				  fifth_factor.lo + sixth_factor.lo +
				  seventh_factor.lo + eighth_factor.lo) *
	coord_range.lo;
      negative_contribution.lo = negative_contribution.hi = 0;
    }

    // If the minimum is negative, maximum is positive,
    else {
      negative_contribution.lo = (first_factor.lo + second_factor.lo +
				  third_factor.lo + fourth_factor.lo + 
				  fifth_factor.lo + sixth_factor.lo +
				  seventh_factor.lo + eighth_factor.lo) *
	coord_range.hi;
      negative_contribution.hi = positive_contribution.lo = 0;
      positive_contribution.hi = (first_factor.lo + second_factor.lo +
				  third_factor.lo + fourth_factor.lo + 
				  fifth_factor.lo + sixth_factor.lo +
				  seventh_factor.lo + eighth_factor.lo) *
	coord_range.lo;
    }

    DEBUG_ASSERT(negative_contribution.hi <= 0 &&
		 negative_contribution.lo <= negative_contribution.hi);
    DEBUG_ASSERT(positive_contribution.lo >= 0 &&
		 positive_contribution.hi >= positive_contribution.lo);
  }

  double AxilrodTellerForceKernelNegativeEvaluate
  (int dimension_in, const double *first_point, const double *second_point,
   double first_distance, double first_distance_pow_three,
   double first_distance_pow_five, double first_distance_pow_seven,
   double second_distance, double second_distance_pow_three,
   double second_distance_pow_five, double third_distance,
   double third_distance_pow_three, double third_distance_pow_five) {
    
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
      (first_distance_pow_five * second_distance_pow_five * third_distance);
    double seventh_factor = (-1.875) /
      (first_distance_pow_seven * second_distance * third_distance_pow_three);
    double eigth_factor = (-1.875) /
      (first_distance_pow_seven * second_distance_pow_three * third_distance);
    
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

  Matrix lower_bound_positive_contributions_;
  
  Matrix upper_bound_positive_contributions_;
  
  Matrix lower_bound_negative_contributions_;

  Matrix upper_bound_negative_contributions_;

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
    lower_bound_squared_distances_.SetZero();
    upper_bound_squared_distances_.Init(order_, order_);
    upper_bound_squared_distances_.SetZero();

    // Allocate the space for storing temporary computation results.
    lower_bound_positive_contributions_.Init(3, 3);    
    upper_bound_positive_contributions_.Init(3, 3);
    lower_bound_negative_contributions_.Init(3, 3);
    upper_bound_negative_contributions_.Init(3, 3);

    // Index permutation stuff...
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

  template<typename Tree, typename TBound>
  void EvaluateHelper
  (const ArrayList<Tree *> &nodes, const Matrix &lower_bound_squared_distances,
   const Matrix &upper_bound_squared_distances) {

    DRange negative_contributions, positive_contributions;

    for(index_t p = 0; p < order_; p++) {

      double *min_positive_contribution =
	lower_bound_positive_contributions_.GetColumnPtr(p);
      double *max_positive_contribution =
	upper_bound_positive_contributions_.GetColumnPtr(p);
      double *max_negative_contribution = 
	upper_bound_negative_contributions_.GetColumnPtr(p);
      double *min_negative_contribution =
	lower_bound_negative_contributions_.GetColumnPtr(p);
      
      const TBound &first_bound = nodes[first_index[p]]->bound();
      const TBound &second_bound = nodes[second_index[p]]->bound();

      DRange distance_first_second, distance_first_second_pow_three,
	distance_first_second_pow_five, distance_first_second_pow_seven,
	distance_first_third, distance_first_third_pow_three,
	distance_first_third_pow_five, distance_second_third,
	distance_second_third_pow_three, distance_second_third_pow_five;

      distance_first_second.lo =
	math::Pow<1, 2>(lower_bound_squared_distances.get(first_index[p],
							  second_index[p]));
      distance_first_second.hi =
	math::Pow<1, 2>(upper_bound_squared_distances.get(first_index[p],
							  second_index[p]));
      distance_first_second_pow_three.lo =
	distance_first_second.lo *
	lower_bound_squared_distances.get(first_index[p], second_index[p]);
      distance_first_second_pow_three.hi =
	distance_first_second.hi *
	upper_bound_squared_distances.get(first_index[p], second_index[p]);
      distance_first_second_pow_five.lo =
	lower_bound_squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_three.lo;
      distance_first_second_pow_five.hi =
	upper_bound_squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_three.hi;
      distance_first_second_pow_seven.lo =
	lower_bound_squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_five.lo;
      distance_first_second_pow_seven.hi =
	upper_bound_squared_distances.get(first_index[p], second_index[p]) *
	distance_first_second_pow_five.hi;
      distance_first_third.lo =
	math::Pow<1, 2>(lower_bound_squared_distances.get(first_index[p],
							  third_index[p]));
      distance_first_third.hi =
	math::Pow<1, 2>(upper_bound_squared_distances.get(first_index[p],
							  third_index[p]));
      distance_first_third_pow_three.lo =
	distance_first_third.lo *
	lower_bound_squared_distances.get(first_index[p], third_index[p]);
      distance_first_third_pow_three.hi =
	distance_first_third.hi *
	upper_bound_squared_distances.get(first_index[p], third_index[p]);
      distance_first_third_pow_five.lo =
	lower_bound_squared_distances.get(first_index[p], third_index[p]) *
	distance_first_third_pow_three.lo;
      distance_first_third_pow_five.hi =
	upper_bound_squared_distances.get(first_index[p], third_index[p]) *
	distance_first_third_pow_three.hi;
      distance_second_third.lo =
	math::Pow<1, 2>(lower_bound_squared_distances.get(second_index[p],
							  third_index[p]));
      distance_second_third.hi =
	math::Pow<1, 2>(upper_bound_squared_distances.get(second_index[p],
							  third_index[p])); 
      distance_second_third_pow_three.lo =
	distance_second_third.lo *
	lower_bound_squared_distances.get(second_index[p], third_index[p]);
      distance_second_third_pow_three.hi =
	distance_second_third.hi *
	upper_bound_squared_distances.get(second_index[p], third_index[p]);
      distance_second_third_pow_five.lo =
	lower_bound_squared_distances.get(second_index[p], third_index[p]) *
	distance_second_third_pow_three.lo;
      distance_second_third_pow_five.hi =
	upper_bound_squared_distances.get(second_index[p], third_index[p]) *
	distance_second_third_pow_three.hi;

      for(index_t counter = 0; counter < dimension_; counter++) {
	AxilrodTellerForceKernelPositiveEvaluate
	  (counter, first_bound, second_bound,
	   distance_first_second,
	   distance_first_second_pow_three,
	   distance_first_second_pow_five,
	   distance_first_second_pow_seven,
	   distance_first_third,
	   distance_first_third_pow_three,
	   distance_first_third_pow_five,
	   distance_second_third,
	   distance_second_third_pow_three,
	   distance_second_third_pow_five, negative_contributions,
	   positive_contributions);
	min_negative_contribution[counter] += negative_contributions.lo;
	max_negative_contribution[counter] += negative_contributions.hi;
	min_positive_contribution[counter] += positive_contributions.lo;
	max_positive_contribution[counter] += positive_contributions.hi;
	  
	AxilrodTellerForceKernelNegativeEvaluate
	  (counter, first_bound, second_bound,
	   distance_first_second,
	   distance_first_second_pow_three,
	   distance_first_second_pow_five,
	   distance_first_second_pow_seven,
	   distance_first_third,
	   distance_first_third_pow_three,
	   distance_first_third_pow_five,
	   distance_second_third,
	   distance_second_third_pow_three,
	   distance_second_third_pow_five, negative_contributions,
	   positive_contributions);
	min_negative_contribution[counter] += negative_contributions.lo;
	max_negative_contribution[counter] += negative_contributions.hi;
	min_positive_contribution[counter] += positive_contributions.lo;
	max_positive_contribution[counter] += positive_contributions.hi;

      } // iterating over each dimension (row-wise)...
    }
  }

  void EvaluateHelper
  (const ArrayList<Matrix *> &sets, const ArrayList<index_t> &indices,
   const Matrix &squared_distances) {

    for(index_t p = 0; p < order_; p++) {

      double *positive_contribution =
	lower_bound_positive_contributions_.GetColumnPtr(p);
      double *negative_contribution = 
	upper_bound_negative_contributions_.GetColumnPtr(p);
      
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

      } // iterating over each dimension (row-wise)...
    }
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

	// Mirror the distances across diagonals, just in case you
	// need it.
	squared_distances.set(second_index, first_index, squared_distance);
      }
    }
  }

  template<typename Tree>
  bool PairwiseEvaluateNodeLoop(ArrayList<Tree *> &nodes,
				Matrix &lower_bound_squared_distances,
				Matrix &upper_bound_squared_distances) {
    
    // Evaluate the squared distance between (first_index, start).
    for(index_t first_index = 0; first_index < order_ - 1; first_index++) {
      for(index_t second_index = first_index + 1; second_index < order_;
	  second_index++) {

	double min_squared_distance =
	  nodes[first_index]->bound().MinDistanceSq
	  (nodes[second_index]->bound());

	if(min_squared_distance == 0) {
	  return true;
	}

	double max_squared_distance =
	  nodes[first_index]->bound().MaxDistanceSq
	  (nodes[second_index]->bound());

	lower_bound_squared_distances.set(first_index, second_index,
					  min_squared_distance);
	upper_bound_squared_distances.set(first_index, second_index,
					  max_squared_distance);

	// Mirror the distances across diagonals, just in case you
	// need it.
	lower_bound_squared_distances.set(second_index, first_index,
					  min_squared_distance);
	upper_bound_squared_distances.set(second_index, first_index,
					  min_squared_distance);
      }
    }
    return false;
  }

  template<typename Tree>
  void ComputeError(ArrayList<Tree *> &nodes,
		    Matrix &negative_force_vector_e,
		    Vector &l1_norm_negative_force_vector_u,
		    Vector &l1_norm_positive_force_vector_l,
		    Matrix &positive_force_vector_e,
		    Vector &n_pruned, Vector &used_error) {
    
    // Temporary variable...
    double min_tmp_contrib, max_tmp_contrib, error, average;

    // Force vector pointers for each node.
    double *negative_force_vector_e_first = negative_force_vector_e.
      GetColumnPtr(0);
    double *negative_force_vector_e_second = negative_force_vector_e.
      GetColumnPtr(1);
    double *negative_force_vector_e_third = negative_force_vector_e.
      GetColumnPtr(2);
    double *positive_force_vector_e_first = positive_force_vector_e.
      GetColumnPtr(0);
    double *positive_force_vector_e_second = positive_force_vector_e.
      GetColumnPtr(1);
    double *positive_force_vector_e_third = positive_force_vector_e.
      GetColumnPtr(2);

    for(index_t i = 0; i < dimension_; i++) {
      
      // Error for positive contributions...
      {
	min_tmp_contrib = lower_bound_positive_contributions_.get(i, 0);
	max_tmp_contrib = upper_bound_positive_contributions_.get(i, 0);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[0] += error;
	l1_norm_negative_force_vector_u[0] += min_tmp_contrib;
	negative_force_vector_e_first[i] += (-average);
	used_error[1] += error;
	l1_norm_positive_force_vector_l[1] += min_tmp_contrib;
	positive_force_vector_e_second[i] += average;
      }

      {
	min_tmp_contrib = lower_bound_positive_contributions_.get(i, 1);
	max_tmp_contrib = upper_bound_positive_contributions_.get(i, 1);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[0] += error;
	l1_norm_negative_force_vector_u[0] += min_tmp_contrib;
	negative_force_vector_e_first[i] += (-average);
	used_error[2] += error;
	l1_norm_positive_force_vector_l[2] += min_tmp_contrib;
	positive_force_vector_e_third[i] += average;
      }

      {
	min_tmp_contrib = lower_bound_positive_contributions_.get(i, 2);
	max_tmp_contrib = upper_bound_positive_contributions_.get(i, 2);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[1] += error;
	l1_norm_negative_force_vector_u[1] += min_tmp_contrib;
	negative_force_vector_e_second[i] += (-average);
	used_error[2] += error;
	l1_norm_positive_force_vector_l[2] += min_tmp_contrib;
	positive_force_vector_e_third[i] += average;
      }
      
      // Check negative contribution accumulations...
      {
	min_tmp_contrib = lower_bound_negative_contributions_.get(i, 0);
	max_tmp_contrib = upper_bound_negative_contributions_.get(i, 0);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[0] += error;
	l1_norm_positive_force_vector_l[0] += (-max_tmp_contrib);
	positive_force_vector_e_first[i] += (-average);
	used_error[1] += error;
	l1_norm_negative_force_vector_u[1] += (-max_tmp_contrib);
	negative_force_vector_e_second[i] += average;
      }

      {
	min_tmp_contrib = lower_bound_negative_contributions_.get(i, 1);
	max_tmp_contrib = upper_bound_negative_contributions_.get(i, 1);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[0] += error;
	l1_norm_positive_force_vector_l[0] += (-max_tmp_contrib);
	positive_force_vector_e_first[i] += (-average);
	used_error[2] += error;
	l1_norm_negative_force_vector_u[2] += (-max_tmp_contrib);
	negative_force_vector_e_third[i] += average;
      }

      {
	min_tmp_contrib = lower_bound_negative_contributions_.get(i, 2);
	max_tmp_contrib = upper_bound_negative_contributions_.get(i, 2);
	error = 0.5 * (max_tmp_contrib - min_tmp_contrib);
	average = 0.5 * (max_tmp_contrib + min_tmp_contrib);
	used_error[1] += error;
	l1_norm_positive_force_vector_l[1] += (-max_tmp_contrib);
	positive_force_vector_e_second[i] += (-average);
	used_error[2] += error;
	l1_norm_negative_force_vector_u[2] += (-max_tmp_contrib);
	negative_force_vector_e_third[i] += average;
      }
    } // end of iterating over each dimension...
  }

  /** @brief Computes the min/max values of each pair-wise kernel
   *         values.
   *
   *  @return false, if any of the minimum distance is zero, since it
   *  would blow up the kernel value, true otherwise.
   */
  template<typename TGlobal, typename Tree>
  bool ComputeFiniteDifference
  (TGlobal &globals, ArrayList<Tree *> &nodes,
   Matrix &negative_force_vector_e, Vector &l1_norm_negative_force_vector_u,
   Vector &l1_norm_positive_force_vector_l, Matrix &positive_force_vector_e,
   Vector &n_pruned, Vector &used_error) {

    // Clear the contribution accumulator.
    lower_bound_positive_contributions_.SetZero();
    upper_bound_positive_contributions_.SetZero();
    lower_bound_negative_contributions_.SetZero();
    upper_bound_negative_contributions_.SetZero();

    // Evaluate the pairwise distances (min and max) among the nodes,
    // but if you get any minimum distance of zero, then don't do it!
    if(PairwiseEvaluateNodeLoop(nodes, lower_bound_squared_distances_,
				upper_bound_squared_distances_)) {
      return false;
    }

    // Evaluate the contributions for each of the particles using the
    // distance bounds.
    EvaluateHelper<Tree, typename Tree::Bound>(nodes,
					       lower_bound_squared_distances_,
					       upper_bound_squared_distances_);

    ComputeError(nodes,
		 negative_force_vector_e, l1_norm_negative_force_vector_u,
		 l1_norm_positive_force_vector_l, positive_force_vector_e,
		 n_pruned, used_error);

    return true;
  }

  /** @brief Evaluate the kernel given the chosen indices exhaustively.
   */
  template<typename Global, typename MultiTreeQueryResult>
  void EvaluateMain(Global &globals, const ArrayList<Matrix *> &sets,
		    MultiTreeQueryResult &query_results) {
    
    // Evaluate the pairwise distances.    
    PairwiseEvaluateLoop(sets, globals.chosen_indices,
			 lower_bound_squared_distances_);

    // Clear the contribution accumulator.
    lower_bound_positive_contributions_.SetZero();
    upper_bound_negative_contributions_.SetZero();

    // Evaluate the positive kernels at (0, 1), (0, 2), (1, 2) and (0,
    // 2), (0, 1), (1, 2) and (1, 2), (0, 2), (0, 1) in dimension 0,
    // 1, 2.
    EvaluateHelper(sets, globals.chosen_indices, 
		   lower_bound_squared_distances_);
    
    // Get the positive/negative force vector for the first, second,
    // third particle in the list.
    double *positive_force_vector_first_particle =
      query_results.positive_force_vector_e.GetColumnPtr
      (globals.chosen_indices[0]);
    double &l1_norm_positive_force_vector_l_first_particle =
      query_results.l1_norm_positive_force_vector_l[globals.chosen_indices[0]];
    double *negative_force_vector_first_particle =
      query_results.negative_force_vector_e.GetColumnPtr
      (globals.chosen_indices[0]);
    double &l1_norm_negative_force_vector_u_first_particle =
      query_results.l1_norm_negative_force_vector_u[globals.chosen_indices[0]];

    double *positive_force_vector_second_particle =
      query_results.positive_force_vector_e.GetColumnPtr
      (globals.chosen_indices[1]);
    double &l1_norm_positive_force_vector_l_second_particle =
      query_results.l1_norm_positive_force_vector_l[globals.chosen_indices[1]];
    double *negative_force_vector_second_particle =
      query_results.negative_force_vector_e.GetColumnPtr
      (globals.chosen_indices[1]);
    double &l1_norm_negative_force_vector_u_second_particle =
      query_results.l1_norm_negative_force_vector_u[globals.chosen_indices[1]];

    double *positive_force_vector_third_particle =
      query_results.positive_force_vector_e.GetColumnPtr
      (globals.chosen_indices[2]);
    double &l1_norm_positive_force_vector_l_third_particle =
      query_results.l1_norm_positive_force_vector_l[globals.chosen_indices[2]];
    double *negative_force_vector_third_particle =
      query_results.negative_force_vector_e.GetColumnPtr
      (globals.chosen_indices[2]);
    double &l1_norm_negative_force_vector_u_third_particle =
      query_results.l1_norm_negative_force_vector_u[globals.chosen_indices[2]];

    // Temporary variable...
    double tmp_contrib;

    for(index_t i = 0; i < dimension_; i++) {
      
      // Check positive contributions...
      if((tmp_contrib = lower_bound_positive_contributions_.get(i, 0)) > 0) {
	negative_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_first_particle += tmp_contrib;
	positive_force_vector_second_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_second_particle += tmp_contrib;
      }
      else {
	positive_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_first_particle += (-tmp_contrib);
	negative_force_vector_second_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_second_particle += (-tmp_contrib);
      }
      if((tmp_contrib = lower_bound_positive_contributions_.get(i, 1)) > 0) {
	negative_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_first_particle += tmp_contrib;
	positive_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_third_particle += tmp_contrib;;
      }
      else {
	positive_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_first_particle += (-tmp_contrib);
	negative_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_third_particle += (-tmp_contrib);
      }
      if((tmp_contrib = lower_bound_positive_contributions_.get(i, 2)) > 0) {
	negative_force_vector_second_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_second_particle += tmp_contrib;
	positive_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_third_particle += tmp_contrib;
      }
      else {
	positive_force_vector_second_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_second_particle += (-tmp_contrib);
	negative_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_third_particle += (-tmp_contrib);
      }
      
      // Check negative contribution accumulations...
      if((tmp_contrib = upper_bound_negative_contributions_.get(i, 0)) > 0) {
	negative_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_first_particle += tmp_contrib;
	positive_force_vector_second_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_second_particle += tmp_contrib;
      }
      else {
	positive_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_first_particle += (-tmp_contrib);
	negative_force_vector_second_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_second_particle += (-tmp_contrib);
      }
      if((tmp_contrib = upper_bound_negative_contributions_.get(i, 1)) > 0) {
	negative_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_first_particle += tmp_contrib;
	positive_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_third_particle += tmp_contrib;
      }
      else {
	positive_force_vector_first_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_first_particle += (-tmp_contrib);
	negative_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_third_particle += (-tmp_contrib);
      }
      if((tmp_contrib = upper_bound_negative_contributions_.get(i, 2)) > 0) {
	negative_force_vector_second_particle[i] += (-tmp_contrib);
	l1_norm_negative_force_vector_u_second_particle += tmp_contrib;
	positive_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_positive_force_vector_l_third_particle += tmp_contrib;
      }
      else {
	positive_force_vector_second_particle[i] += (-tmp_contrib);
	l1_norm_positive_force_vector_l_second_particle += (-tmp_contrib);
	negative_force_vector_third_particle[i] += tmp_contrib;
	l1_norm_negative_force_vector_u_third_particle += (-tmp_contrib);
      }
    } // end of looping over each dimension...
  }
  
};

#endif
