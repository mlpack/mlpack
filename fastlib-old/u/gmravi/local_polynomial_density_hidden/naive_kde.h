/** @file naive_kde.h
 *
 *  This file contains an implementation of the naive KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef NAIVE_KDE_H
#define NAIVE_KDE_H
#define EPSILON pow(10,-10)


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
 *    naive_kde.Init(queries, references, kde_module);
 *
 *    // important to make sure that you don't call Init on results!
 *    naive_kde.Compute(&results);
 *  @endcode
 */
template<typename TKernel>
class NaiveKde {
  
  FORBID_ACCIDENTAL_COPIES(NaiveKde);

 private:

  ////////// Private Member Variables //////////

  /** @brief Pointer to the module holding the parameters. */
  //struct datanode *module_;

  /** @brief The column-oriented query dataset. */
  Matrix qset_;
  
  /** @brief The column-oriented reference dataset. */
  Matrix rset_;
  
  /** @brief The kernel function. */
  TKernel kernel_;

  /** @brief The computed densities. */
  Vector densities_;
  
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
  void get_density_estimates(Vector &results) { 

    results.CopyValues(densities_);
  
   
  }

  ////////// User-level Functions //////////

  /** @brief Compute kernel density estimates naively after intialization
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */

  void Compare(Matrix &foreign_density, Vector &boundary_points){
    
    //For each query point find out the R.M.S.E, and for the boundary
    //points find out the R.M.S.E
    
    double mean_sqd_error=0;
    double boundary_mean_sqd_error=0;
    index_t number_of_boundary_points=0;
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      mean_sqd_error+=pow(densities_[q]-foreign_density.get(0,q),2);
      
      //If dataset is one dimensional
      if(qset_.n_rows()==1){
	
	if(qset_.get(0,q)< boundary_points[1]||  
	   qset_.get(0,q)>boundary_points[2]){
	  
	  boundary_mean_sqd_error+=pow(foreign_density.get(0,q)-
				       densities_[q],2);  
	  number_of_boundary_points++;
	}
      }
    }
    
    mean_sqd_error/=qset_.n_cols();
    boundary_mean_sqd_error/=number_of_boundary_points;
    
    double root_mean_sqd_error=sqrt(mean_sqd_error);
    double boundary_root_mean_sqd_error=sqrt(boundary_mean_sqd_error);
    
    printf("RESULTS FOR NAIVE KDE.............\n");
    printf("Total root mean squared error is %f\n",root_mean_sqd_error);
    printf("Boundary root mean squared error is %f\n",
	   boundary_root_mean_sqd_error);
    printf("Number of boundary points are %d\n",number_of_boundary_points);
  }


  void Compute(Vector *results) {

    printf("\nStarting naive KDE...\n");
    //fx_timer_start(module_, "naive_kde_compute");

    // compute unnormalized sum
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);

	if(fabs(dsqd)>EPSILON){	
	  densities_[q] += kernel_.EvalUnnormOnSq(dsqd);
	}
      }
    }
    
    // then normalize it
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) * 
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
    //fx_timer_stop(module_, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");

   
  }

  /** @brief Compute kernel density estimates naively after intialization
   */
  void Compute() {

    printf("\nStarting naive KDE...\n");
    //fx_timer_start(module_, "naive_kde_compute");

    // compute unnormalized sum
   
    for(index_t q = 0; q < qset_.n_cols(); q++) {      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) 
	{
	  const double *r_col = rset_.GetColumnPtr(r);

	  double dsqd =
	  la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col); 
	  
	  densities_[q] += kernel_.EvalUnnormOnSq(dsqd);
	}
    }
    
    // then normalize it
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) * 
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
    //fx_timer_stop(module_, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");
  }

  /** @brief Initialize the naive KDE algorithm object with the query and the
   *         reference datasets and the parameter list.
   *
   *  @param qset The column-oriented query dataset.
   *  @param rset The column-oriented reference dataset.
   *  @param module_in The module holding the parameters.
   */
  void Init(Matrix &qset, Matrix &rset, double bandwidth) {

    // set the datanode module to be the incoming one
    //module_ = module_in;

    // get datasets

    qset_.Init(qset.n_rows(),qset.n_cols());
    rset_.Init(rset.n_rows(),rset.n_cols());

    qset_.CopyValues(qset);
    rset_.CopyValues(rset);

    // get bandwidth
    kernel_.Init(bandwidth);

    printf("Bandwidth for naive kde calc is %f\n",bandwidth);
    
    // allocate density storage
    densities_.Init(qset.n_cols());
    densities_.SetZero();
  }

  /** @brief Output KDE results to a stream 
   *
   *  If the user provided "--naive_kde_output=" argument, then the
   *  output will be directed to a file whose name is provided after
   *  the equality sign.  Otherwise, it will be provided to the
   *  screen.
   */
  /*void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if(fx_param_exists(module_, "naive_kde_output")) {
      fname = fx_param_str(module_, "naive_kde_output", NULL);
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
    }*/
  
  /** @brief Computes the maximum relative error for the approximated
   *         density estimates.
   *
   *  The maximum relative error is output after the program finishes
   *  the run under /maximum_relative_error_for_fast_KDE/
   *
   *  @param density_estimates The vector holding approximated density
   *                           estimates.
   */
  void ComputeMaximumRelativeError(const Vector &density_estimates) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < densities_.length(); q++) {
      double rel_err = fabs(density_estimates[q] - densities_[q]) / 
	densities_[q];

      if(isnan(density_estimates[q]) || isinf(density_estimates[q]) || 
	 isnan(densities_[q]) || isinf(densities_[q])) {
	VERBOSE_MSG("Warning: Got infs or nans!\n");
      }

      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    //fx_format_result(module_, "maximum_relative_error_for_fast_KDE", "%g", 
    //max_rel_err);
  }

};

#endif
