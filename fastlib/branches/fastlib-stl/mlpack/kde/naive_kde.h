/** @file naive_kde.h
 *
 *  This file contains an implementation of the naive KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef NAIVE_KDE_H
#define NAIVE_KDE_H

#include "mlpack/allknn/allknn.h"

/** @brief A templatized class for computing the KDE naively.
 *
 *  This class is only intended to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    NaiveKde naive_kde;
 *    struct datanode* kde_module;
 *    Vector results;
 *
 *    kde_module = fx_submodule(NULL, "kde", "kde_module");
 *    naive_kde.Init(queries, references, reference_weights, kde_module);
 *
 *    // Important to make sure that you don't call Init on results!
 *    naive_kde.Compute(&results);
 *  @endcode
 */
template<typename TKernel>
class NaiveKde {
  
 private:

  ////////// Private Member Variables //////////

  /** @brief Pointer to the module holding the parameters. */
  struct datanode *module_;

  /** @brief The column-oriented query dataset. */
  arma::mat qset_;
  
  /** @brief The column-oriented reference dataset. */
  arma::mat rset_;

  /** @brief The vector containing the reference set weights. */
  arma::vec rset_weights_;

  /** @brief The kernel function. */
  std::vector<TKernel> kernels_;

  /** @brief The computed densities. */
  arma::vec densities_;

  /** @brief The normalizing constant. */
  double norm_const_;

 public:

  ////////// Constructor/Destructor //////////

  /** @brief Constructor - does not do anything */
  NaiveKde() { }

  /** @brief Destructor - does not do anything */
  ~NaiveKde() { }

  ////////// Getters/Setters //////////

  /** @brief Get the density estimate 
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */
  void get_density_estimates(arma::vec& results) { 
    results = densities_;
  }

  ////////// User-level Functions //////////

  /** @brief Compute kernel density estimates naively after intialization
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */
  void Compute(arma::vec& results) {

    printf("\nStarting naive KDE...\n");
    fx_timer_start(module_, "naive_kde_compute");

    for(index_t q = 0; q < qset_.n_cols; q++) {
      
      arma::vec q_col = qset_.unsafe_col(q);

      // Compute unnormalized sum first.
      for(index_t r = 0; r < rset_.n_cols; r++) {
	arma::vec r_col = rset_.unsafe_col(r);
	double dsqd = la::DistanceSqEuclidean(q_col, r_col);
	
	densities_[q] += rset_weights_[r] * kernels_[r].EvalUnnormOnSq(dsqd);
      }

      // Then normalize it.
      densities_[q] /= norm_const_;
    }
    fx_timer_stop(module_, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");

    // retrieve density estimates
    get_density_estimates(results);
  }

  /** @brief Compute kernel density estimates naively after intialization
   */
  void Compute() {

    printf("\nStarting naive KDE...\n");
    fx_timer_start(module_, "naive_kde_compute");

    for(index_t q = 0; q < qset_.n_cols; q++) {
      
      arma::vec q_col = qset_.unsafe_col(q);
      
      // Compute unnormalized sum.
      for(index_t r = 0; r < rset_.n_cols; r++) {
	arma::vec r_col = rset_.unsafe_col(r);
	double dsqd = la::DistanceSqEuclidean(q_col, r_col);
	
	densities_[q] += rset_weights_[r] * kernels_[r].EvalUnnormOnSq(dsqd);
      }
      // Then, normalize it.
      densities_[q] /= norm_const_;
    }
    fx_timer_stop(module_, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");
  }

  void Init(arma::mat& qset, arma::mat& rset, struct datanode *module_in) {

    // Use the uniform weights for a moment.
    arma::mat uniform_weights(1, rset.n_cols);
    uniform_weights.fill(1.0);

    Init(qset, rset, uniform_weights, module_in);
  }

  /** @brief Initialize the naive KDE algorithm object with the query and the
   *         reference datasets and the parameter list.
   *
   *  @param qset The column-oriented query dataset.
   *  @param rset The column-oriented reference dataset.
   *  @param module_in The module holding the parameters.
   */
  void Init(arma::mat& qset, arma::mat& rset, arma::mat& reference_weights,
	    struct datanode *module_in) {

    // Set the datanode module to be the incoming one.
    module_ = module_in;

    // Get datasets.
    qset_ = qset;
    rset_ = rset;
    rset_weights_.set_size(reference_weights.n_elem);
    for(index_t i = 0; i < rset_weights_.n_elem; i++) {
      rset_weights_[i] = reference_weights(0, i);
    }    

    // Compute the normalizing constant.
    double weight_sum = 0;
    for(index_t i = 0; i < rset_weights_.n_elem; i++) {
      weight_sum += rset_weights_[i];
    }
    
    // Get bandwidth and compute the normalizing constant.
    kernels_.reserve(rset_.n_cols);
    if(!strcmp(fx_param_str(module_, "mode", "variablebw"), "variablebw")) {

      // Initialize the kernels for each reference point.
      int knns = fx_param_int_req(module_, "knn");
      AllkNN all_knn;
      all_knn.Init(&rset_, 20, knns);
      arma::Col<index_t> resulting_neighbors;
      arma::vec squared_distances;    
      
      fx_timer_start(fx_root, "bandwidth_initialization");
      all_knn.ComputeNeighbors(resulting_neighbors, squared_distances);
      
      for(index_t i = 0; i < squared_distances.n_elem; i += knns) {
	kernels_[i / knns].Init(sqrt(squared_distances[i + knns - 1]));
      }
      fx_timer_stop(fx_root, "bandwidth_initialization");

      // Renormalize the reference weights according to the bandwidths
      // that have been chosen.
      double min_norm_const = DBL_MAX;
      for(index_t i = 0; i < rset_weights_.n_elem; i++) {
	double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows);
	min_norm_const = std::min(min_norm_const, norm_const);
      }
      for(index_t i = 0; i < rset_weights_.n_elem; i++) {
	double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows);
	rset_weights_[i] *= (min_norm_const / norm_const);
      }
      
      // Compute normalization constant.
      norm_const_ = weight_sum * min_norm_const;
    }
    else {
      for(index_t i = 0; i < kernels_.size(); i++) {
	kernels_[i].Init(fx_param_double_req(module_, "bandwidth"));
      }
      norm_const_ = kernels_[0].CalcNormConstant(qset_.n_rows) * weight_sum;
    }

    // Allocate density storage.
    densities_.zeros(qset.n_cols);
  }

  /** @brief Output KDE results to a stream 
   *
   *  If the user provided "--naive_kde_output=" argument, then the
   *  output will be directed to a file whose name is provided after
   *  the equality sign.  Otherwise, it will be provided to the
   *  screen.
   */
  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    {
      fname = fx_param_str(module_, "naive_kde_output", 
			   "naive_kde_output.txt");
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols; q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }
  
  /** @brief Computes the maximum relative error for the approximated
   *         density estimates.
   *
   *  The maximum relative error is output after the program finishes
   *  the run under /maximum_relative_error_for_fast_KDE/
   *
   *  @param density_estimates The vector holding approximated density
   *                           estimates.
   */
  void ComputeMaximumRelativeError(const arma::vec& density_estimates) {
    
    double max_rel_err = 0;
    FILE *stream = fopen("relative_error_output.txt", "w+");
    int within_limit = 0;

    for(index_t q = 0; q < densities_.n_elem; q++) {
      double rel_err = (fabs(density_estimates[q] - densities_[q]) <
			DBL_EPSILON) ?
	0:fabs(density_estimates[q] - densities_[q]) / 
	densities_[q];

      if(std::isnan(density_estimates[q]) || std::isinf(density_estimates[q]) || 
         std::isnan(densities_[q]) || std::isinf(densities_[q])) {
	VERBOSE_MSG(0,"Warning: Got infs or nans!\n");
      }

      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
      if(rel_err <= fx_param_double(module_, "relative_error", 0.01)) {
	within_limit++;
      }

      fprintf(stream, "%g\n", rel_err);
    }
    
    fclose(stream);
    fx_format_result(module_, "maximum_relative_error_for_fast_KDE", "%g", 
		     max_rel_err);
    fx_format_result(module_, "under_relative_error_limit", "%d",
		     within_limit);
  }

};

#endif
