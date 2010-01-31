/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#ifndef DUALTREE_KDE_CV_COMMON_H
#define DUALTREE_KDE_CV_COMMON_H

#include "inverse_normal_cdf.h"

class DualtreeKdeCVCommon {

 public:
  
  template<typename TTree, typename TAlgorithm>
  static bool MonteCarloPrunable
  (TTree *qnode, TTree *rnode, double probability, DRange &dsqd_range,
   DRange &first_kernel_value_range, DRange &second_kernel_value_range,
   double &first_dl, double &first_de, double &first_du, 
   double &first_used_error, double &second_dl, double &second_de, 
   double &second_du, double &second_used_error, double &delta_n_pruned,
   TAlgorithm *kde_object) {
    
    // If there are too few pairs, then return.
    if(qnode->count() * rnode->count() < 50) {
      return false;
    }
    
    // Temporary kernel sums.
    double first_kernel_sum = 0, second_kernel_sum = 0;
    double first_squared_kernel_sum = 0, second_squared_kernel_sum = 0;
    
    // Commence sampling...
    double standard_score = 
      InverseNormalCDF::Compute(probability + 0.5 * (1 - probability));
    
    // The initial number of samples is equal to the default.
    int num_samples = 50;
    int total_samples = 0;
    
    for(index_t s = 0; s < num_samples; s++) {
      
      index_t random_query_point_index =
	math::RandInt(qnode->begin(), qnode->end());
      index_t random_reference_point_index = 
	math::RandInt(rnode->begin(), rnode->end());
      
      // Get the pointer to the current query point.
      const double *query_point = 
	(kde_object->rset_).GetColumnPtr(random_query_point_index);
      
      // Get the pointer to the current reference point.
      const double *reference_point = 
	(kde_object->rset_).GetColumnPtr(random_reference_point_index);
      
      // Compute the pairwise distance and kernel value.
      double squared_distance = la::DistanceSqEuclidean
	((kde_object->rset_).n_rows(), query_point, reference_point);
      
      double first_weighted_kernel_value;
      double second_weighted_kernel_value;
      kde_object->EvalUnnormOnSq_
	(random_reference_point_index, squared_distance, 
	 &first_weighted_kernel_value, &second_weighted_kernel_value);
      first_kernel_sum += first_weighted_kernel_value;
      second_kernel_sum += second_weighted_kernel_value;
      first_squared_kernel_sum += math::Sqr(first_weighted_kernel_value);
      second_squared_kernel_sum += math::Sqr(second_weighted_kernel_value);

    } // end of taking samples...
    
    // Increment total number of samples.
    total_samples += num_samples;
    
    // Compute the current estimate of the sample mean and the sample
    // variance.
    double first_sample_mean = first_kernel_sum / ((double) total_samples);
    double first_sample_variance =
      (first_squared_kernel_sum - total_samples * first_sample_mean * 
       first_sample_mean) / math::Sqr((double) total_samples - 1);
    double second_sample_mean = second_kernel_sum / ((double) total_samples);
    double second_sample_variance =
      (second_squared_kernel_sum - total_samples * second_sample_mean *
       second_sample_mean) / math::Sqr((double) total_samples - 1);

    // Refine the lower bound using the new lower bound info.
    double first_mass_l_change = qnode->count() * 
      rnode->stat().get_weight_sum() *
      (first_sample_mean - standard_score * sqrt(first_sample_variance));
    double first_new_mass_l = (kde_object->first_sum_l_) + first_mass_l_change;
    double second_mass_l_change = qnode->count() *
      rnode->stat().get_weight_sum() *
      (second_sample_mean - standard_score * sqrt(second_sample_variance));
    double second_new_mass_l = (kde_object-> second_sum_l_) +
      second_mass_l_change;
    
    // Compute the allowed error.
    double proportion = 1.0 / (kde_object->rroot_->count() * 
			       kde_object->rroot_->stat().get_weight_sum() -
			       kde_object->n_pruned_);
    double first_allowed_err = 
      (kde_object->relative_error_ * first_new_mass_l - 
       kde_object->first_used_error_) * proportion;
    double second_allowed_err =
      (kde_object->relative_error_ * second_new_mass_l - 
       kde_object->second_used_error_) * proportion;
        
    if(sqrt(first_sample_variance) * standard_score <= first_allowed_err &&
       sqrt(second_sample_variance) * standard_score <= second_allowed_err) {
      first_dl = std::max(first_dl, first_mass_l_change);
      first_de = qnode->count() * rnode->stat().get_weight_sum() * 
	first_sample_mean;
      first_used_error = qnode->count() * rnode->stat().get_weight_sum() * 
	standard_score * sqrt(first_sample_variance);
      second_dl = std::max(second_dl, second_mass_l_change);
      second_de = qnode->count() * rnode->stat().get_weight_sum() * 
	second_sample_mean;
      second_used_error = qnode->count() * rnode->stat().get_weight_sum() * 
	standard_score * sqrt(second_sample_variance);
      return true;
    }
    else {
      return false;
    }
  }
  
  template<typename TTree, typename TAlgorithm>
  static bool Prunable(TTree *qnode, TTree *rnode, double probability, 
		       DRange &dsqd_range, DRange &first_kernel_value_range,
		       DRange &second_kernel_value_range,
		       double &first_dl, double &first_de, double &first_du, 
		       double &first_used_error, 
		       double &second_dl, double &second_de, double &second_du,
		       double &second_used_error, double &delta_n_pruned,
		       TAlgorithm *kde_object) {
    
    // the new lower bound after incorporating new info
    first_dl = first_kernel_value_range.lo * qnode->count() * 
      rnode->stat().get_weight_sum();
    first_de = 0.5 * qnode->count() * rnode->stat().get_weight_sum() * 
      (first_kernel_value_range.lo + first_kernel_value_range.hi);
    first_du = (first_kernel_value_range.hi - 1) * qnode->count() *
      rnode->stat().get_weight_sum();
    second_dl = second_kernel_value_range.lo * qnode->count() *
      rnode->stat().get_weight_sum();
    second_de = 0.5 * qnode->count() * rnode->stat().get_weight_sum() * 
      (second_kernel_value_range.lo + second_kernel_value_range.hi);
    second_du = (second_kernel_value_range.hi - 1) * qnode->count() * 
      rnode->stat().get_weight_sum();
   
    // Refine the lower bound using the new lower bound info.
    double first_new_mass_l = (kde_object->first_sum_l_) + first_dl;
    double second_new_mass_l = (kde_object-> second_sum_l_) + second_dl;
    
    // Compute the allowed error.
    double proportion = 
      (qnode->count() * rnode->stat().get_weight_sum()) /
      (kde_object->rroot_->count() * 
       kde_object->rroot_->stat().get_weight_sum() - kde_object->n_pruned_);
    double first_allowed_err = 
      (kde_object->relative_error_ * first_new_mass_l - 
       kde_object->first_used_error_) *
      proportion;
    double second_allowed_err =
      (kde_object->relative_error_ * second_new_mass_l - 
       kde_object->second_used_error_) *
      proportion;

    // This is error per each query/reference pair for a fixed query
    double first_kernel_diff = 
      0.5 * (first_kernel_value_range.hi - first_kernel_value_range.lo);
    double second_kernel_diff =
      0.5 * (second_kernel_value_range.hi - second_kernel_value_range.lo);
    
    // this is total error for each query point
    first_used_error = first_kernel_diff * qnode->count() * 
      rnode->stat().get_weight_sum();
    second_used_error = second_kernel_diff * qnode->count() *
      rnode->stat().get_weight_sum();
    
    // number of reference points for possible pruning.
    delta_n_pruned = qnode->count() * rnode->stat().get_weight_sum();
    
    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    return (!isnan(first_allowed_err)) && (!isnan(second_allowed_err)) &&
      (first_used_error <= first_allowed_err) &&
      (second_used_error <= second_allowed_err);
  }
};

#endif
