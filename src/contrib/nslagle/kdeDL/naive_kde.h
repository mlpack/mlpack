/** @file naive_kde.h
 *
 *  This file contains an implementation of the naive KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef NAIVE_KDE_H
#define NAIVE_KDE_H

#include "mlpack/methods/neighbor_search/neighbor_search.h"

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
  
  //FORBID_ACCIDENTAL_COPIES(NaiveKde);

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
  NaiveKde() {    
  }

  /** @brief Destructor - does not do anything */
  ~NaiveKde() {
  }

  ////////// Getters/Setters //////////

  /** @brief Get the density estimate 
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */
  void get_density_estimates(arma::vec *results) { 
    *results = arma::vec(densities_.size());
    
    for(size_t i = 0; i < densities_.size(); i++) {
      (*results)[i] = densities_[i];
    }
  }

  ////////// User-level Functions //////////

  /** @brief Compute kernel density estimates naively after intialization
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */
  void Compute(arma::vec *results) {

    printf("\nStarting naive KDE...\n");
    CLI::StartTimer("naive_kde_compute");

    for(size_t q = 0; q < qset_.n_cols(); q++) {
      
      const arma::vec q_col = qset_.unsafe_col(q);

      // Compute unnormalized sum first.
      for(size_t r = 0; r < rset_.n_cols(); r++) {
	const arma::vec r_col = rset_.unsafe_col(r);
	double dsqd = kernel::LMetric<2,false>::Evaluate(q_col, r_col);
	
	densities_[q] += rset_weights_[r] * kernels_[r].EvalUnnormOnSq(dsqd);
      }

      // Then normalize it.
      densities_[q] /= norm_const_;
    }
    CLI::StopTimer("naive_kde_compute");
    printf("\nNaive KDE completed...\n");

    // retrieve density estimates
    get_density_estimates(results);
  }

  /** @brief Compute kernel density estimates naively after intialization
   */
  void Compute() {

    printf("\nStarting naive KDE...\n");
    CLI::StartTimer("naive_kde_compute");

    for(size_t q = 0; q < qset_.n_cols(); q++) {
      
      const arma::vec q_col = qset_.unsafe_col(q);
      
      // Compute unnormalized sum.
      for(size_t r = 0; r < rset_.n_cols(); r++) {
	const arma::vec r_col = rset_.unsafe_col(r);
	double dsqd = kernel::LMetric<2,false>::Evaluate (q_col, r_col);
	
	densities_[q] += rset_weights_[r] * kernels_[r].EvalUnnormOnSq(dsqd);
      }
      // Then, normalize it.
      densities_[q] /= norm_const_;
    }
    CLI::StopTimer("naive_kde_compute");
    printf("\nNaive KDE completed...\n");
  }

  void Init(arma::mat &qset, arma::mat &rset, struct datanode *module_in) {

    // Use the uniform weights for a moment.
    arma::mat uniform_weights(1, rset.n_cols());
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
  void Init(arma::mat &qset, arma::mat &rset, arma::mat &reference_weights,
	    struct datanode *module_in) {

    // Set the datanode module to be the incoming one.
    module_ = module_in;

    // Get datasets.
    qset_ = arma::mat(qset.n_rows, qset.n_cols);
    for (size_t c = 0; c < qset.n_cols; ++c)
    {
      for (size_t r = 0; r < qset.n_rows; ++r)
      {
        qset_(r,c) = qset(r,c);
      }
    }
    rset_ = arma::mat(rset.n_rows, rset.n_cols);
    for (size_t c = 0; c < rset.n_cols; ++c)
    {
      for (size_t r = 0; r < rset.n_rows; ++r)
      {
        rset_(r,c) = rset(r,c);
      }
    }
    rset_weights_ = arma::vec(reference_weights.n_cols());
    for(size_t i = 0; i < rset_weights_.size(); i++)
    {
      rset_weights_[i] = reference_weights(0, i);
    }

    // Compute the normalizing constant.
    double weight_sum = 0;
    for(size_t i = 0; i < rset_weights_.size(); i++) {
      weight_sum += rset_weights_[i];
    }

    // Get bandwidth and compute the normalizing constant.
    kernels_.Init(rset_.n_cols());
    if(!strcmp(CLI::GetParam<std::string>("mode").c_str(), "variablebw")) {

      // Initialize the kernels for each reference point.
      int knns = CLI::GetParam<int>("knn");
      AllkNN all_knn = AllkNN(rset_, 20, knns);
      arma::Mat<size_t> resulting_neighbors;
      arma::mat squared_distances;

      CLI::StartTimer("bandwidth_initialization");
      all_knn.ComputeNeighbors(resulting_neighbors, squared_distances);

      for(size_t i = 0; i < squared_distances.size(); i += knns) {
	kernels_[i / knns].Init(sqrt(squared_distances[i + knns - 1]));
      }
      CLI::StopTimer("bandwidth_initialization");

      // Renormalize the reference weights according to the bandwidths
      // that have been chosen.
      double min_norm_const = DBL_MAX;
      for(size_t i = 0; i < rset_weights_.size(); i++) {
	double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows());
	min_norm_const = std::min(min_norm_const, norm_const);
      }
      for(size_t i = 0; i < rset_weights_.size(); i++) {
	double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows());
	rset_weights_[i] *= (min_norm_const / norm_const);
      }
      
      // Compute normalization constant.
      norm_const_ = weight_sum * min_norm_const;
    }
    else {
      for(size_t i = 0; i < kernels_.size(); i++) {
	kernels_[i].Init(CLI::GetParam<double>("bandwidth"));
      }
      norm_const_ = kernels_[0].CalcNormConstant(qset_.n_rows()) * weight_sum;
    }

    // Allocate density storage.
    densities_ = arma::vec(qset.n_cols());
    densities_.zeros();
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
      fname = CLI::GetParam<std::string>("naive_kde_output");
      stream = fopen(fname, "w+");
    }
    for(size_t q = 0; q < qset_.n_cols(); q++) {
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
  void ComputeMaximumRelativeError(const arma::vec &density_estimates) {
    
    double max_rel_err = 0;
    FILE *stream = fopen("relative_error_output.txt", "w+");
    int within_limit = 0;

    for(size_t q = 0; q < densities_.size(); q++) {
      double rel_err = (fabs(density_estimates[q] - densities_[q]) <
			DBL_EPSILON) ?
	0:fabs(density_estimates[q] - densities_[q]) / 
	densities_[q];      

      if(isnan(density_estimates[q]) || isinf(density_estimates[q]) || 
	 isnan(densities_[q]) || isinf(densities_[q])) {
        Log::Info << "Warning: Got infs or nans!";
      }

      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
      if(rel_err <= CLI::GetParam<double>("relative_error")) {
	within_limit++;
      }

      fprintf(stream, "%g\n", rel_err);
    }
    
    fclose(stream);
    //fx_format_result(module_, "maximum_relative_error_for_fast_KDE", "%g", 
		//     max_rel_err);
    //fx_format_result(module_, "under_relative_error_limit", "%d",
		//     within_limit);
  }

};

#endif
