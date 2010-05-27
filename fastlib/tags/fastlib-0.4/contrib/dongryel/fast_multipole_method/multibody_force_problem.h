#ifndef AXILROD_TELLER_FORCE_H
#define AXILROD_TELLER_FORCE_H

#include "fastlib/fastlib.h"

#include "mlpack/series_expansion/kernel_aux.h"

#include "multibody_kernel.h"
#include "../multitree_template/multitree_utility.h"

class AxilrodTellerForceProblem {

 public:

  class MultiTreeDelta {

   public:

    Matrix negative_force_vector_l;

    Matrix negative_force_vector_e;

    Matrix negative_force_vector_u;

    Vector l1_norm_negative_force_vector_u;

    Vector l1_norm_positive_force_vector_l;

    Matrix positive_force_vector_l;

    Matrix positive_force_vector_e;

    Matrix positive_force_vector_u;

    Vector n_pruned;
    
    Vector used_error;

    Vector probabilistic_used_error;

    OT_DEF_BASIC(MultiTreeDelta) {
      OT_MY_OBJECT(negative_force_vector_l);
      OT_MY_OBJECT(negative_force_vector_e);
      OT_MY_OBJECT(negative_force_vector_u);
      OT_MY_OBJECT(l1_norm_negative_force_vector_u);
      OT_MY_OBJECT(l1_norm_positive_force_vector_l);
      OT_MY_OBJECT(positive_force_vector_l);
      OT_MY_OBJECT(positive_force_vector_e);
      OT_MY_OBJECT(positive_force_vector_u);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(used_error);
      OT_MY_OBJECT(probabilistic_used_error);
    }

   public:

    template<typename TGlobal, typename Tree>
    void ComputeMonteCarloEstimates(TGlobal &globals,
				    const ArrayList<Matrix *> &sets,
				    ArrayList<Tree *> &nodes,
				    const Vector &total_n_minus_one_tuples) {

      // Clear the deterministic error component.
      used_error.SetZero();

      // If any of the distance evaluation resulted in zero minimum
      // distance, then return false.
      globals.kernel_aux.ComputeMonteCarloEstimates
	(globals, sets, nodes, total_n_minus_one_tuples,
	 negative_force_vector_l, negative_force_vector_e,
	 negative_force_vector_u, l1_norm_negative_force_vector_u,
	 l1_norm_positive_force_vector_l, positive_force_vector_l,
	 positive_force_vector_e, positive_force_vector_u, n_pruned,
	 probabilistic_used_error);
      
    }

    template<typename TGlobal, typename Tree>
    bool ComputeFiniteDifference(TGlobal &globals,
				 ArrayList<Tree *> &nodes,
				 const Vector &total_n_minus_one_tuples) {

      // If any of the distance evaluation resulted in zero minimum
      // distance, then return false.
      bool flag = globals.kernel_aux.ComputeFiniteDifference
	(globals, nodes, total_n_minus_one_tuples, negative_force_vector_l,
	 negative_force_vector_e, negative_force_vector_u,
	 l1_norm_negative_force_vector_u, l1_norm_positive_force_vector_l,
	 positive_force_vector_l, positive_force_vector_e,
	 positive_force_vector_u, n_pruned, used_error);

      return flag;
    }

    void SetZero() {
      negative_force_vector_l.SetZero();
      negative_force_vector_e.SetZero();
      negative_force_vector_u.SetZero();
      l1_norm_negative_force_vector_u.SetZero();
      l1_norm_positive_force_vector_l.SetZero();
      positive_force_vector_l.SetZero();
      positive_force_vector_e.SetZero();
      positive_force_vector_u.SetZero();
      used_error.SetZero();
      probabilistic_used_error.SetZero();
    }

    void Init(const Vector &total_n_minus_one_tuples) {

      // Hard-codes to use 3-dimensional vectors.
      negative_force_vector_l.Init(3, 3);
      negative_force_vector_e.Init(3, 3);
      negative_force_vector_u.Init(3, 3);
      l1_norm_negative_force_vector_u.Init(3);
      l1_norm_positive_force_vector_l.Init(3);
      positive_force_vector_l.Init(3, 3);
      positive_force_vector_e.Init(3, 3);
      positive_force_vector_u.Init(3, 3);
      n_pruned.Init(3);
      used_error.Init(3);
      probabilistic_used_error.Init(3);

      // Copy the number of pruned tuples...
      n_pruned.CopyValues(total_n_minus_one_tuples);

      // Initializes to zeros...
      SetZero();
    }    
  };

  class MultiTreeQueryPostponed {
    
   public:

    Vector negative_force_vector_l;

    Vector negative_force_vector_e;

    Vector negative_force_vector_u;

    double l1_norm_negative_force_vector_u;

    double l1_norm_positive_force_vector_l;

    Vector positive_force_vector_l;

    Vector positive_force_vector_e;

    Vector positive_force_vector_u;

    double n_pruned;
    
    double used_error;

    double probabilistic_used_error;

    void ApplyDelta(const MultiTreeDelta &delta_in, index_t node_index) {
      la::AddTo(3, delta_in.negative_force_vector_l.GetColumnPtr(node_index),
		negative_force_vector_l.ptr());
      la::AddTo(3, delta_in.negative_force_vector_e.GetColumnPtr(node_index),
		negative_force_vector_e.ptr());
      la::AddTo(3, delta_in.negative_force_vector_u.GetColumnPtr(node_index),
		negative_force_vector_u.ptr());
      l1_norm_negative_force_vector_u += 
	delta_in.l1_norm_negative_force_vector_u[node_index];
      l1_norm_positive_force_vector_l +=
	delta_in.l1_norm_positive_force_vector_l[node_index];
      la::AddTo(3, delta_in.positive_force_vector_l.GetColumnPtr(node_index),
		positive_force_vector_l.ptr());
      la::AddTo(3, delta_in.positive_force_vector_e.GetColumnPtr(node_index),
		positive_force_vector_e.ptr());
      la::AddTo(3, delta_in.positive_force_vector_u.GetColumnPtr(node_index),
		positive_force_vector_u.ptr());
      n_pruned += delta_in.n_pruned[node_index];
      used_error += delta_in.used_error[node_index];
      probabilistic_used_error = 
	sqrt(math::Sqr(probabilistic_used_error) +
	     math::Sqr(delta_in.probabilistic_used_error[node_index]));
    }

    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
      la::AddTo(3, postponed_in.negative_force_vector_l.ptr(),
		negative_force_vector_l.ptr());
      la::AddTo(3, postponed_in.negative_force_vector_e.ptr(),
		negative_force_vector_e.ptr());
      la::AddTo(3, postponed_in.negative_force_vector_u.ptr(),
		negative_force_vector_u.ptr());
      l1_norm_negative_force_vector_u +=
	postponed_in.l1_norm_negative_force_vector_u;
      l1_norm_positive_force_vector_l +=
	postponed_in.l1_norm_positive_force_vector_l;
      la::AddTo(3, postponed_in.positive_force_vector_l.ptr(),
		positive_force_vector_l.ptr());
      la::AddTo(3, postponed_in.positive_force_vector_e.ptr(),
		positive_force_vector_e.ptr());
      la::AddTo(3, postponed_in.positive_force_vector_u.ptr(),
		positive_force_vector_u.ptr());
      n_pruned += postponed_in.n_pruned;
      used_error += postponed_in.used_error;     
      probabilistic_used_error =
	sqrt(math::Sqr(probabilistic_used_error) +
	     math::Sqr(postponed_in.probabilistic_used_error));
    }

    void SetZero() {
      negative_force_vector_l.SetZero();
      negative_force_vector_e.SetZero();
      negative_force_vector_u.SetZero();
      l1_norm_negative_force_vector_u = 0;
      l1_norm_positive_force_vector_l = 0;
      positive_force_vector_l.SetZero();
      positive_force_vector_e.SetZero();
      positive_force_vector_u.SetZero();
      n_pruned = 0;
      used_error = 0;
      probabilistic_used_error = 0;
    }

    void Init() {

      // Hard-codes to use 3-dimensional vectors.
      negative_force_vector_l.Init(3);
      negative_force_vector_e.Init(3);
      negative_force_vector_u.Init(3);
      positive_force_vector_l.Init(3);
      positive_force_vector_e.Init(3);
      positive_force_vector_u.Init(3);

      // Initializes to zeros...
      SetZero();
    }
  };

  class MultiTreeQuerySummary {
   public:

    double l1_norm_negative_force_vector_u;
    
    double l1_norm_positive_force_vector_l;
    
    double n_pruned_l;
    
    double used_error_u;

    double probabilistic_used_error_u;

    OT_DEF_BASIC(MultiTreeQuerySummary) {
      OT_MY_OBJECT(l1_norm_negative_force_vector_u);
      OT_MY_OBJECT(l1_norm_positive_force_vector_l);
      OT_MY_OBJECT(n_pruned_l);
      OT_MY_OBJECT(used_error_u);
      OT_MY_OBJECT(probabilistic_used_error_u);
    }

   public:

    template<typename TQueryResult>
    void Accumulate(const TQueryResult &query_results, index_t q_index) {
      l1_norm_negative_force_vector_u =
	std::min(l1_norm_negative_force_vector_u,
		 fabs(query_results.negative_force_vector_e.get(0, q_index) +
		      query_results.negative_force_vector_e.get(1, q_index) + 
		      query_results.negative_force_vector_e.get(2, q_index)));
      l1_norm_positive_force_vector_l =
	std::min(l1_norm_positive_force_vector_l,
		 query_results.positive_force_vector_e.get(0, q_index) +
		 query_results.positive_force_vector_e.get(1, q_index) +
		 query_results.positive_force_vector_e.get(2, q_index));
      n_pruned_l =
	std::min(n_pruned_l, query_results.n_pruned[q_index]);
      used_error_u =
	std::max(used_error_u, query_results.used_error[q_index]);
      probabilistic_used_error_u =
	std::max(probabilistic_used_error_u,
		 query_results.probabilistic_used_error[q_index]);
    }
    
    template<typename TGlobal, typename TQueryResult>
    void PostAccumulate(TGlobal &globals, const TQueryResult &query_results,
			index_t first, index_t count) {

    }

    void SetZero() {
      l1_norm_negative_force_vector_u = 0;
      l1_norm_positive_force_vector_l = 0;
      n_pruned_l = 0;
      used_error_u = 0;
      probabilistic_used_error_u = 0;
    }
    
    void ApplyDelta(const MultiTreeDelta &delta_in, index_t delta_index) {
      
      l1_norm_negative_force_vector_u +=
	delta_in.l1_norm_negative_force_vector_u[delta_index];
      l1_norm_positive_force_vector_l +=
	delta_in.l1_norm_positive_force_vector_l[delta_index];      
    }

    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
      l1_norm_negative_force_vector_u += 
	postponed_in.l1_norm_negative_force_vector_u;
      l1_norm_positive_force_vector_l +=
	postponed_in.l1_norm_positive_force_vector_l;
      n_pruned_l += postponed_in.n_pruned;
      used_error_u += postponed_in.used_error;
      probabilistic_used_error_u =
	sqrt(math::Sqr(probabilistic_used_error_u) +
	     math::Sqr(postponed_in.probabilistic_used_error));
    }

    void Accumulate(const MultiTreeQuerySummary &summary_in) {
      l1_norm_negative_force_vector_u = 
	std::min(l1_norm_negative_force_vector_u,
		 summary_in.l1_norm_negative_force_vector_u);
      l1_norm_positive_force_vector_l = 
	std::min(l1_norm_positive_force_vector_l,
		 summary_in.l1_norm_positive_force_vector_l);
      n_pruned_l = std::min(n_pruned_l, summary_in.n_pruned_l);
      used_error_u = std::max(used_error_u, summary_in.used_error_u);
      probabilistic_used_error_u = 
	std::max(probabilistic_used_error_u,
		 summary_in.probabilistic_used_error_u);
    }

    void StartReaccumulate() {
      l1_norm_negative_force_vector_u = DBL_MAX;
      l1_norm_positive_force_vector_l = DBL_MAX;
      n_pruned_l = DBL_MAX;
      used_error_u = 0;
      probabilistic_used_error_u = 0;
    }

  };

  class MultiTreeQueryStat {

   public:

    MultiTreeQueryPostponed postponed;
    
    MultiTreeQuerySummary summary;
    
    double priority;
    
    Vector mean;
    
    index_t count;
    
    bool in_strata;

    double num_precomputed_tuples;

    OT_DEF_BASIC(MultiTreeQueryStat) {
      OT_MY_OBJECT(postponed);
      OT_MY_OBJECT(summary);
      OT_MY_OBJECT(priority);
      OT_MY_OBJECT(mean);
      OT_MY_OBJECT(count);
      OT_MY_OBJECT(in_strata);
      OT_MY_OBJECT(num_precomputed_tuples);
    }
    
   public:

    double SumOfPerDimensionVariances
    (const Matrix &dataset, index_t &start, index_t &count) {

      double total_variance = 0;
      for(index_t i = start; i < start + count; i++) {
	const double *point = dataset.GetColumnPtr(i);
	for(index_t d = 0; d < 3; d++) {
	  total_variance += math::Sqr(point[d] - mean[d]);
	}
      }
      total_variance /= ((double) count);
      return total_variance;
    }

    void FinalPush(MultiTreeQueryStat &child_stat) {
      child_stat.postponed.ApplyPostponed(postponed);
    }
    
    void SetZero() {
      postponed.SetZero();
      summary.SetZero();
      priority = 0;
      mean.SetZero();
      in_strata = false;
      num_precomputed_tuples = 0;
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count_in) {
      postponed.Init();
      mean.Init(3);
      SetZero();
      count = count_in;

      // Compute the mean vector.
      for(index_t i = start; i < start + count; i++) {
	const double *point = dataset.GetColumnPtr(i);
	la::AddTo(3, point, mean.ptr());
      }
      la::Scale(3, 1.0 / ((double) count), mean.ptr());

      // Compute the priority of this node which is basically the
      // number of points times sum of per-dimension variances.
      double sum_of_per_dimension_variances = SumOfPerDimensionVariances
	(dataset, start, count);
      priority = count * sum_of_per_dimension_variances;
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count_in,
	      const MultiTreeQueryStat& left_stat, 
	      const MultiTreeQueryStat& right_stat) {
      postponed.Init();
      mean.Init(3);
      SetZero();
      count = count_in;

      la::ScaleOverwrite(left_stat.count, left_stat.mean, &mean);
      la::AddExpert(3, right_stat.count, right_stat.mean.ptr(), mean.ptr());
      la::Scale(3, 1.0 / ((double) count), mean.ptr());

      // Compute the priority of this node which is basically the
      // number of points times sum of per-dimension variances.
      double sum_of_per_dimension_variances = SumOfPerDimensionVariances
	(dataset, start, count);
      priority = count * sum_of_per_dimension_variances;
    }
    
    template<typename TKernelAux>
    void Init(const TKernelAux &kernel_aux_in) {
    }
    
    template<typename TBound, typename TKernelAux>
    void Init(const TBound &bounding_primitive,
	      const TKernelAux &kernel_aux_in) {
      
      // Reset the postponed quantities to zero.
      SetZero();
    }
  };

  class MultiTreeReferenceStat {
  };

  class MultiTreeQueryResult {
   public:

    Vector l1_norm_positive_force_vector_l;

    Matrix positive_force_vector_l;

    /** @brief Each column is a force vector for each particle.
     */    
    Matrix positive_force_vector_e;

    Matrix positive_force_vector_u;
    
    Matrix negative_force_vector_l;

    Matrix negative_force_vector_e;

    Matrix negative_force_vector_u;

    Vector l1_norm_negative_force_vector_u;

    Matrix final_results_l;

    Matrix final_results;

    Matrix final_results_u;

    Vector n_pruned;
    
    Vector used_error;

    Vector probabilistic_used_error;

    /** @brief The boolean map specifying whether each particle
     *         satisfies the coordinate-wise relative error bound
     *         using the lower and the upper bounds.
     */
    GenVector<bool> relative_error_satisfied;

    /** @brief The number of finite-difference prunes.
     */
    int num_finite_difference_prunes;
    
    /** @brief The number of Monte Carlo prunes.
     */
    int num_monte_carlo_prunes;

    OT_DEF_BASIC(MultiTreeQueryResult) {
      OT_MY_OBJECT(l1_norm_positive_force_vector_l);
      OT_MY_OBJECT(positive_force_vector_l);
      OT_MY_OBJECT(positive_force_vector_e);
      OT_MY_OBJECT(positive_force_vector_u);
      OT_MY_OBJECT(negative_force_vector_l);
      OT_MY_OBJECT(negative_force_vector_e);
      OT_MY_OBJECT(negative_force_vector_u);
      OT_MY_OBJECT(l1_norm_negative_force_vector_u);
      OT_MY_OBJECT(final_results_l);
      OT_MY_OBJECT(final_results);
      OT_MY_OBJECT(final_results_u);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(used_error);
      OT_MY_OBJECT(probabilistic_used_error);
      OT_MY_OBJECT(relative_error_satisfied);
      OT_MY_OBJECT(num_finite_difference_prunes);
      OT_MY_OBJECT(num_monte_carlo_prunes);
    }

   public:

    void MaximumRelativeError(const MultiTreeQueryResult &other_results,
			      double *max_relative_error,
			      double *negative_max_relative_error,
			      double *positive_max_relative_error) {

      // Compute the net force...
      Vector net_force, net_force_for_approx;
      net_force.Init(3);
      net_force_for_approx.Init(3);
      net_force.SetZero();
      net_force_for_approx.SetZero();

      FILE *relative_error_output = fopen("relative_error.txt", "w+");
      *max_relative_error = 0;
      *negative_max_relative_error = 0;
      *positive_max_relative_error = 0;

      // Number of approximated force vectors within the relative
      // error.
      index_t within_relative_error = 0;

      for(index_t i = 0; i < used_error.length(); i++) {

	// Add up the net force...
	la::AddTo(3, final_results.GetColumnPtr(i), net_force.ptr());
	la::AddTo(3, other_results.final_results.GetColumnPtr(i),
		  net_force_for_approx.ptr());

	double l1_norm_error = 
	  la::RawLMetric<1>(final_results.n_rows(),
			    final_results.GetColumnPtr(i),
			    other_results.final_results.GetColumnPtr(i));
	double l1_norm_exact = 0;
	const double *exact_vector = final_results.GetColumnPtr(i);
	for(index_t d = 0; d < final_results.n_rows(); d++) {
	  l1_norm_exact += fabs(exact_vector[d]);
	}
	
	fprintf(relative_error_output, "%g ", l1_norm_error / l1_norm_exact);

	if(l1_norm_error > l1_norm_exact * AxilrodTellerForceProblem::
	   relative_error_) {
	  fprintf(relative_error_output, "X ");
	}

	*max_relative_error = std::max(*max_relative_error,
				       l1_norm_error / l1_norm_exact);

	if(l1_norm_error <= l1_norm_exact * 
	   AxilrodTellerForceProblem::relative_error_) {
	  within_relative_error++;
	}

        double positive_l1_norm_error =
          la::RawLMetric<1>(positive_force_vector_e.n_rows(),
                            positive_force_vector_e.GetColumnPtr(i),
                            other_results.positive_force_vector_e.GetColumnPtr
			    (i));
        double positive_l1_norm_exact = 0;
        const double *positive_exact_vector = 
	  positive_force_vector_e.GetColumnPtr(i);
        for(index_t d = 0; d < positive_force_vector_e.n_rows(); d++) {
          positive_l1_norm_exact += fabs(positive_exact_vector[d]);
        }
	
	double negative_l1_norm_error =
	  la::RawLMetric<1>(negative_force_vector_e.n_rows(),
			    negative_force_vector_e.GetColumnPtr(i),
                            other_results.negative_force_vector_e.GetColumnPtr
			    (i));
	double negative_l1_norm_exact = 0;
        const double *negative_exact_vector =
          negative_force_vector_e.GetColumnPtr(i);
        for(index_t d = 0; d < negative_force_vector_e.n_rows(); d++) {
          negative_l1_norm_exact += fabs(negative_exact_vector[d]);
        }

	fprintf(relative_error_output, "%g ", positive_l1_norm_error /
		positive_l1_norm_exact);
	*positive_max_relative_error = std::max(*positive_max_relative_error,
						positive_l1_norm_error /
						positive_l1_norm_exact);
	fprintf(relative_error_output, "%g\n", negative_l1_norm_error /
		negative_l1_norm_exact);
	*negative_max_relative_error = std::max(*negative_max_relative_error,
						negative_l1_norm_error /
						negative_l1_norm_exact);
      }
      net_force.PrintDebug("Net force: ", stdout);
      net_force_for_approx.PrintDebug("Net force approximated: ", stdout);

      printf("Within relative error: %d\n", within_relative_error);

      fclose(relative_error_output);
    }

    template<typename Tree>
    void UpdatePrunedComponents(const ArrayList<Tree *> &reference_nodes,
				index_t q_index) {
    }

    void FinalPush(const Matrix &qset, 
		   const MultiTreeQueryStat &stat_in, index_t q_index) {
      
      ApplyPostponed(stat_in.postponed, q_index);
    }

    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in, 
			index_t q_index) {
      
      l1_norm_positive_force_vector_l[q_index] += 
	postponed_in.l1_norm_positive_force_vector_l;
      la::AddTo(3, postponed_in.positive_force_vector_l.ptr(),
		positive_force_vector_l.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.positive_force_vector_e.ptr(),
		positive_force_vector_e.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.positive_force_vector_u.ptr(),
		positive_force_vector_u.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.negative_force_vector_l.ptr(),
		negative_force_vector_l.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.negative_force_vector_e.ptr(),
		negative_force_vector_e.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.negative_force_vector_u.ptr(),
		negative_force_vector_u.GetColumnPtr(q_index));
      l1_norm_negative_force_vector_u[q_index] +=
	postponed_in.l1_norm_negative_force_vector_u;
      n_pruned[q_index] += postponed_in.n_pruned;
      used_error[q_index] += postponed_in.used_error;
      probabilistic_used_error[q_index] =
	sqrt(math::Sqr(probabilistic_used_error[q_index]) +
	     math::Sqr(postponed_in.probabilistic_used_error));
    }

    void Init(int num_queries) {
      l1_norm_positive_force_vector_l.Init(num_queries);
      positive_force_vector_l.Init(3, num_queries);
      positive_force_vector_e.Init(3, num_queries);
      positive_force_vector_u.Init(3, num_queries);
      negative_force_vector_l.Init(3, num_queries);
      negative_force_vector_e.Init(3, num_queries);
      negative_force_vector_u.Init(3, num_queries);
      l1_norm_negative_force_vector_u.Init(num_queries);
      final_results_l.Init(3, num_queries);
      final_results.Init(3, num_queries);
      final_results_u.Init(3, num_queries);
      n_pruned.Init(num_queries);
      used_error.Init(num_queries);
      probabilistic_used_error.Init(num_queries);
      relative_error_satisfied.Init(num_queries);
      
      SetZero();
    }

    template<typename MultiTreeGlobal>
    void PostProcess(const MultiTreeGlobal &globals, index_t q_index) {

      la::AddOverwrite(3, positive_force_vector_l.GetColumnPtr(q_index),
		       negative_force_vector_l.GetColumnPtr(q_index),
		       final_results_l.GetColumnPtr(q_index));
      la::AddOverwrite(3, positive_force_vector_e.GetColumnPtr(q_index),
		       negative_force_vector_e.GetColumnPtr(q_index),
		       final_results.GetColumnPtr(q_index));
      la::AddOverwrite(3, positive_force_vector_u.GetColumnPtr(q_index),
		       negative_force_vector_u.GetColumnPtr(q_index),
		       final_results_u.GetColumnPtr(q_index));
    }

    template<typename MultiTreeGlobal>
    void Finalize(const MultiTreeGlobal &globals,
		  const ArrayList<index_t> &mapping) {
      MultiTreeUtility::ShuffleAccordingToQueryPermutation
	(final_results_l, mapping);
      MultiTreeUtility::ShuffleAccordingToQueryPermutation
	(final_results, mapping);
      MultiTreeUtility::ShuffleAccordingToQueryPermutation
	(final_results_u, mapping);

      // Check whether each particle satisfies the relative error
      // bound (roughly in the probabilistic sense) by examining the
      // lower and the upper bounds in each dimension.
      for(index_t i = 0; i < final_results_l.n_cols(); i++) {
	
	// The lower bound column vector and the upper bound column
	// vector.
	const double *force_vector_l_column = final_results_l.GetColumnPtr(i);
	const double *force_vector_u_column = final_results_u.GetColumnPtr(i);
	
	double coord_diff = 
	  fabs(force_vector_u_column[0] - force_vector_l_column[0]) +
	  fabs(force_vector_u_column[1] - force_vector_l_column[1]) +
	  fabs(force_vector_u_column[2] - force_vector_l_column[2]);
	double l1_norm_lower_bound = 
	  fabs(force_vector_l_column[0]) + fabs(force_vector_l_column[1]) +
	  fabs(force_vector_l_column[2]);

	// For each dimension...
	if(coord_diff > l1_norm_lower_bound *
	   AxilrodTellerForceProblem::relative_error_) {
	  
	  relative_error_satisfied[i] = false;
	}
      }
    }

    void PrintDebug(const char *output_file_name) const {

      FILE *stream = fopen(output_file_name, "w+");
      FILE *relative_error_satisfied_output =
	fopen("relative_error_satisfied_guess.txt", "w+");
      int relative_error_satisfied_guess = 0;

      for(index_t q = 0; q < final_results.n_cols(); q++) {

	const double *force_vector_l_column =
	  final_results_l.GetColumnPtr(q);
	const double *force_vector_e_column = 
	  final_results.GetColumnPtr(q);
	const double *force_vector_u_column =
	  final_results_u.GetColumnPtr(q);

	// Print the lower bound on the coordinate force vector.
	fprintf(stream, "[ ");
	for(index_t d = 0; d < 3; d++) {
	  fprintf(stream, "%g ", force_vector_l_column[d]);
	}
	fprintf(stream, " ] [ ");
	// Print the estimates on the coordinate force vector.
	for(index_t d = 0; d < 3; d++) {
	  fprintf(stream, "%g ", force_vector_e_column[d]);
	}
	fprintf(stream, " ] [ ");
	// Print the upper bound on the coordinate force vector.
	for(index_t d = 0; d < 3; d++) {
	  fprintf(stream, "%g ", force_vector_u_column[d]);
	}
	fprintf(stream, " ] ");
	
	fprintf(stream, " %g\n", n_pruned[q]);
	fprintf(relative_error_satisfied_output, "%d\n", 
		relative_error_satisfied[q]);
	
	if(relative_error_satisfied[q]) {
	  relative_error_satisfied_guess++;
	}
      }
      
      printf("I think %d particles satisfy the relative error...\n",
	     relative_error_satisfied_guess);
      fclose(stream);
      fclose(relative_error_satisfied_output);
    }

    void SetZero() {
      l1_norm_positive_force_vector_l.SetZero();
      positive_force_vector_l.SetZero();
      positive_force_vector_e.SetZero();
      positive_force_vector_u.SetZero();
      negative_force_vector_l.SetZero();
      negative_force_vector_e.SetZero();
      negative_force_vector_u.SetZero();
      l1_norm_negative_force_vector_u.SetZero();
      final_results_l.SetZero();
      final_results.SetZero();
      final_results_u.SetZero();
      n_pruned.SetZero();
      used_error.SetZero();
      probabilistic_used_error.SetZero();

      // By default, I assume all queries satisfy the relative error...
      relative_error_satisfied.SetAll(true);

      num_finite_difference_prunes = 0;
      num_monte_carlo_prunes = 0;
    }
  };

  /** @brief Defines the global variable for the Axilrod-Teller force
   *         computation.
   */
  class MultiTreeGlobal {

   public:
    
    /** @brief The module holding the parameters.
     */
    struct datanode *module;

    /** @brief The kernel object.
     */
    AxilrodTellerForceKernelAux kernel_aux;

    /** @brief The chosen indices.
     */
    ArrayList<index_t> hybrid_node_chosen_indices;

    ArrayList<index_t> query_node_chosen_indices;
    
    ArrayList<index_t> reference_node_chosen_indices;

    /** @brief The total number of 3-tuples that contain a particular
     *         particle.
     */
    double total_n_minus_one_tuples;

   public:

    void Init(index_t total_num_particles, index_t dimension_in,
	      const ArrayList<Matrix *> &reference_targets,
	      struct datanode *module_in) {

      kernel_aux.Init();
      hybrid_node_chosen_indices.Init(AxilrodTellerForceProblem::order);
      
      total_n_minus_one_tuples = 
	math::BinomialCoefficient(total_num_particles - 1,
				  AxilrodTellerForceProblem::order - 1);

      // Set the incoming module for referring to parameters.
      module = module_in;
    }

  };

  /** @brief The order of interaction is 3-tuple problem.
   */
  static const int order = 3;

  static const int num_hybrid_sets = 3;
  
  static const int num_query_sets = 0;

  static const int num_reference_sets = 0;

  static const double relative_error_ = 0.1;

  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleExact(MultiTreeGlobal &globals,
				 MultiTreeQueryResult &results,
				 MultiTreeDelta &delta,
				 const ArrayList<Matrix *> &query_sets,
				 const ArrayList<Matrix *> &reference_sets,
				 const ArrayList<Matrix *> &reference_targets,
				 ArrayList<HybridTree *> &hybrid_nodes,
				 ArrayList<QueryTree *> &query_nodes,
				 ArrayList<ReferenceTree *> &reference_nodes,
				 double total_num_tuples,
				 double total_n_minus_one_tuples_root,
				 Vector &total_n_minus_one_tuples) {

    // Compute delta change for each node...
    if(hybrid_nodes[0] == hybrid_nodes[1] && 
       hybrid_nodes[1] == hybrid_nodes[2]) {

      for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
	total_n_minus_one_tuples[i] -= 
	  (hybrid_nodes[i]->stat().num_precomputed_tuples);
      }
    }

    delta.Init(total_n_minus_one_tuples);

    if(unlikely(hybrid_nodes[0]->stat().in_strata && 
		hybrid_nodes[0] == hybrid_nodes[1] &&
		hybrid_nodes[1] == hybrid_nodes[2])) {
      return true;
    }

    if(!delta.ComputeFiniteDifference(globals, hybrid_nodes,
				      total_n_minus_one_tuples)) {
      return false;
    }
    
    // Consider each node in turn whether it can be pruned or not.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {

      // Refine the summary statistics from the new info...
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	AxilrodTellerForceProblem::MultiTreeQuerySummary new_summary;
	new_summary.InitCopy(hybrid_nodes[i]->stat().summary);
	new_summary.ApplyPostponed(hybrid_nodes[i]->stat().postponed);
	new_summary.ApplyDelta(delta, i);

	double sum = new_summary.l1_norm_negative_force_vector_u +
	  new_summary.l1_norm_positive_force_vector_l;
	
	if((AxilrodTellerForceProblem::relative_error_ * sum -
	    (new_summary.used_error_u + 
	     new_summary.probabilistic_used_error_u)) * 
	   total_n_minus_one_tuples[i] < 
	   delta.used_error[i] * 
	   (total_n_minus_one_tuples_root - new_summary.n_pruned_l)) {

	  /*
	  if(((sum + delta.used_error[i]) - sum) >
	     sum * AxilrodTellerForceProblem::relative_error_) {
	    return false;
	  }
	  */

	  const double *negative_force_vector_e = 
	    delta.negative_force_vector_e.GetColumnPtr(i);
	  const double *positive_force_vector_e =
	    delta.positive_force_vector_e.GetColumnPtr(i);
	  double change_l1_norm = 
	    fabs(negative_force_vector_e[0] + negative_force_vector_e[1] +
		 negative_force_vector_e[2] +
		 positive_force_vector_e[0] + positive_force_vector_e[1] +
		 positive_force_vector_e[2]);
	    
	  if(change_l1_norm *
	     (total_n_minus_one_tuples_root - new_summary.n_pruned_l) >
	     sum * AxilrodTellerForceProblem::relative_error_ *
	     total_n_minus_one_tuples[i]) {
	    return false;
	  }
	}
      }
    }

    // In this case, add the delta contributions to the postponed
    // slots of each node.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	hybrid_nodes[i]->stat().postponed.ApplyDelta(delta, i);
      }
    }
    
    results.num_finite_difference_prunes++;
    return true;
  }

  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleProbabilistic
  (MultiTreeGlobal &globals, MultiTreeQueryResult &results,
   MultiTreeDelta &exact_delta, const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &sets, ArrayList<HybridTree *> &hybrid_nodes,
   ArrayList<QueryTree *> &query_nodes,
   ArrayList<ReferenceTree *> &reference_nodes,
   double total_num_tuples, double total_n_minus_one_tuples_root,
   const Vector &total_n_minus_one_tuples) {
    

    /*
    if(total_num_tuples < 40) {
      return false;
    }

    // Compute delta change for each node...
    MultiTreeDelta delta;
    delta.Init(total_n_minus_one_tuples);
    delta.ComputeMonteCarloEstimates(globals, sets, hybrid_nodes,
				     total_n_minus_one_tuples);

    // Consider each node in turn whether it can be pruned or not.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {

      // Refine the summary statistics from the new info...
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	AxilrodTellerForceProblem::MultiTreeQuerySummary new_summary;
	new_summary.InitCopy(hybrid_nodes[i]->stat().summary);
	new_summary.ApplyPostponed(hybrid_nodes[i]->stat().postponed);
	new_summary.ApplyDelta(delta, i);
	
	// Compute the L1 norm of the positive component and the
	// negative component.
	double sum = new_summary.l1_norm_negative_force_vector_u +
	  new_summary.l1_norm_positive_force_vector_l;

	if((AxilrodTellerForceProblem::relative_error_ * sum -
	    (new_summary.used_error_u +
	     new_summary.probabilistic_used_error_u)) * 
	   total_n_minus_one_tuples[i] <=
	   delta.probabilistic_used_error[i] *
	   (total_n_minus_one_tuples_root - new_summary.n_pruned_l)) {
	  
          if(((sum+ delta.probabilistic_used_error[i]) - 
	      sum) >=
	     sum * AxilrodTellerForceProblem::relative_error_) {
            return false;
          }

	}
      }
    }

    // In this case, add the delta contributions to the postponed
    // slots of each node.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	hybrid_nodes[i]->stat().postponed.ApplyDelta(delta, i);
      }
    }
    
    results.num_monte_carlo_prunes++;
    return true;
    */
    return false;
  }
    
  static void HybridNodeEvaluateMain(MultiTreeGlobal &globals,
				     const ArrayList<Matrix *> &query_sets,
				     const ArrayList<Matrix *> &sets,
				     const ArrayList<Matrix *> &targets,
				     MultiTreeQueryResult &query_results) {
    
    globals.kernel_aux.EvaluateMain(globals, sets, query_results);
  }

  static void ReferenceNodeEvaluateMain(MultiTreeGlobal &globals,
					const ArrayList<Matrix *> &query_sets,
					const ArrayList<Matrix *> &sets,
					const ArrayList<Matrix *> &targets,
					MultiTreeQueryResult &query_results) {
    
  }

};

#endif
