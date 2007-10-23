#ifndef FGT_KDE_H
#define FGT_KDE_H

#include <math.h>
#include <values.h>

/** 
 * Computing kernel estimate using Fast Gauss Transform by Lelie Greengard
 * and John Strain
 */
class FGTKde {
  
 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  GaussianKernel kernel_;

  /** computed densities */
  Vector densities_;
  
  /** accuracy parameter */
  double tau_;

  /*
  void gafexp_(dym *q, dym *d, dyv_array *e, double delta, int nterms,
	       int nallbx, ivec *nsides, dyv *sidelengths, dyv *mincoords,
	       dyv_array *locexp, int nfmax, int nlmax, int kdis, 
	       dyv_array *center, ivec_array *queries_assigned,
	       ivec_array *references_assigned, dyv_array *mcoeffs) {
    int dim = dym_cols(d);

    int totalnumcoeffs = (int) pow(nterms, dim);

    // Step 1: Assign query points and reference points to boxes.
    assign_(q, d, nallbx, nsides, sidelengths, mincoords, center, 
	    queries_assigned, references_assigned);

    // Initialize local expansions to zero.
    int i, j, k, l;

    // Process all reference boxes
    for(i = 0; i < nallbx; i++) {

      ivec *reference_rows = ivec_array_ref(references_assigned, i);
      int ninbox = ivec_size(reference_rows);

      // If the box contains no reference points, skip it.
      if(ninbox <= 0) {
	continue;
      }

      // In this case, no far-field expansion is created
      else if(ninbox <= nfmax) {

	// Get the query boxes that are in the interaction range.
	ivec *nbors = mknbor_(i, nsides, kdis);
	int nnbors = ivec_size(nbors);

	for(j = 0; j < nnbors; j++) {

	  // Number of query points in this neighboring box.
	  int query_box_num = ivec_ref(nbors, j);
	  ivec *query_rows = ivec_array_ref(queries_assigned, query_box_num);
	  int ninnbr = ivec_size(query_rows);
	
	  if(ninnbr <= nlmax) {

	    // Direct Interaction
	    for(k = 0; k < ninnbr; k++) {
	    
	      int query_row = ivec_ref(query_rows, k);
	    
	      for(l = 0; l < ninbox; l++) {
		int reference_row = ivec_ref(reference_rows, l);

		if(query_row == reference_row)
		  continue;

		double dsqd = row_metric_dsqd(q, d, NULL, query_row, 
					      reference_row);
		double pot = exp(-dsqd / delta);

		// Here, I hard-coded to do only one bandwidth.
		dyv_increment(dyv_array_ref(e, query_row), 0, pot);
	      }
	    }
	  
	  }

	  // In this case, take each reference point and convert into the 
	  // Taylor series.
	  else {
	  
	    directLocalAccumulation2(d, reference_rows, query_box_num,
				     locexp, delta,
				     dyv_array_ref(center, query_box_num),
				     nterms, totalnumcoeffs);
	  }
	
	}
      }

      // In this case, create a far field expansion.
      else {

	computeMultipoleCoeffs2(d, mcoeffs ,dim, nterms,
				totalnumcoeffs, i, delta, reference_rows,
				dyv_array_ref(center, i));

	// Get the query boxes that are in the interaction range.
	ivec *nbors = mknbor_(i, nsides, kdis);      
	int nnbors = ivec_size(nbors);

	for(j = 0; j < nnbors; j++) {
	  int query_box_num = ivec_ref(nbors, j);
	  ivec *query_rows = ivec_array_ref(queries_assigned, query_box_num);
	  int ninnbr = ivec_size(query_rows);

	  // If this is true, evaluate far field expansion at each query point.
	  if(ninnbr <= nlmax) {

	    evaluateMultipoleExpansion2(q, query_rows, nterms,
					totalnumcoeffs, mcoeffs,
					i, e, delta, 
					dyv_array_ref(center, i));
	  }

	  // In this case do multipole to local translation.
	  else {

	    translateMultipoleToLocal2(i, query_box_num,
				       mcoeffs, locexp,
				       nterms, totalnumcoeffs, delta,
				       dyv_array_ref(center, i),
				       dyv_array_ref(center, query_box_num));
	  
	  }
	}
      }
    }
    gaeval_(q, e, delta, nterms, nallbx, nsides, locexp, nlmax,
	    queries_assigned, center, totalnumcoeffs);
  }

  void gauss_t(dym *q, dym *d, dyv_array *e, double delta, int nterms, 
	       int nallbx, ivec *nsides, dyv *sidelengths, dyv *mincoords,
	       dyv_array *locexp, dyv_array *center,
	       ivec_array *queries_assigned, ivec_array *references_assigned,
	       dyv_array *mcoeffs) {
    int dim = dym_cols(q);
    int kdis = (int) (sqrt(log(Tau) * -2.0) + 1);

    // This is a slight modification of Strain's cutoff since he never
    // implemented this above 2 dimensions.
    int nfmax = (int) pow(nterms, dim - 1) + 2;
    int nlmax = nfmax;

    // Call gafexp to create all expansions on grid, evaluate all appropriate
    // far-field expansions and evaluate all appropriate direct interactions.
    gafexp_(q, d, e, delta, nterms, nallbx, nsides, *sidelengths, mincoords,
	    locexp, nfmax, nlmax, kdis, center, queries_assigned,
	    references_assigned, mcoeffs);
  }
  */

 public:

  FGTKde() {}
  
  ~FGTKde() {}
  
  // getters and setters
  
  /** get the reference dataset */
  Matrix &get_reference_dataset() { return rset_; }

  /** get the query dataset */
  Matrix &get_query_dataset() { return qset_; }

  /** get the density estimate */
  const Vector &get_density_estimates() { return densities_; }

  void Init(Matrix &qset, Matrix &rset) {
    
    printf("Initializing FGT KDE...\n");
    fx_timer_start(NULL, "fgt_kde_init");

    // initialize the kernel and read in the number of grid points
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // set aliases to the query and reference datasets and initialize
    // query density sets
    qset_.Alias(qset);
    densities_.Init(qset_.n_cols());
    rset_.Alias(rset);

    fx_timer_stop(NULL, "fgt_kde_init");
    printf("FGT KDE initialization completed...\n");
  }

  void Init() {

    const char *rfname = fx_param_str_req(NULL, "data");
    const char *qfname = fx_param_str(NULL, "query", rfname);

    // initialize the kernel and read in the number of grid points
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // read reference dataset
    Dataset ref_dataset;
    ref_dataset.InitFromFile(rfname);
    rset_.Own(&(ref_dataset.matrix()));

    // read query dataset if different
    if(!strcmp(qfname, rfname)) {
      qset_.Alias(rset_);
    }
    else {
      Dataset query_dataset;
      query_dataset.InitFromFile(qfname);
      qset_.Own(&(query_dataset.matrix()));
    }

    printf("Initializing FGT KDE...\n");
    fx_timer_start(NULL, "fgt_kde_init");


    
    fx_timer_stop(NULL, "fgt_kde_init");
    printf("FGT KDE initialization completed...\n");

  }

  void FastGaussTransformPreprocess(double *interaction_radius, 
				    ArrayList<int> &nsides, 
				    Vector &sidelengths, Vector &mincoords, 
				    int *nboxes, int *nterms) {
  
    /**
     * Compute the interaction radius.
     */
    double bandwidth = sqrt(kernel_.bandwidth_sq());
    *interaction_radius = sqrt(-2.0 * kernel_.bandwidth_sq() * log(tau_));

    int di, n, num_rows = rset_.n_cols();
    int dim = rset_.n_rows();

    /**
     * Discretize the grid space into boxes.
     */
    Vector maxcoords;
    maxcoords.Init(dim);
    maxcoords.SetAll(MINDOUBLE);
    double boxside = -1.0;
    *nboxes = 1;

    for(di = 0; di < dim; di++) {
      mincoords[di] = MAXDOUBLE;
    }
    
    for(n = 0; n < num_rows; n++) {
      for(di = 0; di < dim; di++) {
	if(mincoords[di] > rset_.get(di, n)) {
	  mincoords[di] = rset_.get(di, n);
	}
	if(maxcoords[di] < rset_.get(di, n)) {
	  maxcoords[di] = rset_.get(di, n);
	}
      }
    }

    /**
     * Figure out how many boxes lie along each direction.
     */
    for(di = 0; di < dim; di++) {
      nsides[di] = (int) 
	((maxcoords[di] - mincoords[di]) / bandwidth + 1);
      (*nboxes) = (*nboxes) * nsides[di];
      double tmp = (maxcoords[di] - mincoords[di]) /
	(nsides[di] * 2 * bandwidth);
    
      if(tmp > boxside) {
	boxside = tmp;
      }

      sidelengths[di] = (maxcoords[di] - mincoords[di]) / 
	((double) nsides[di]);
      
    }
    
    int ip = 0;
    double two_r = 2.0 * boxside;
    double one_minus_two_r = 1.0 - two_r;
    double ret = 1.0 / pow(one_minus_two_r * one_minus_two_r, dim);
    double factorialvalue = 1.0;
    double r_raised_to_p_alpha = 1.0;
    double first_factor, second_factor;
    double ret2;
                                                                       
    do {
      ip++;
      factorialvalue *= ip;
    
      r_raised_to_p_alpha *= two_r;
      first_factor = 1.0 - r_raised_to_p_alpha;
      first_factor *= first_factor;
      second_factor = r_raised_to_p_alpha * (2.0 - r_raised_to_p_alpha)
	/ sqrt(factorialvalue);
    
      ret2 = ret * (pow((first_factor + second_factor), dim) -
		    pow(first_factor, dim));
    
    } while(ret2 > tau_);

    *nterms = ip;
  }

  void Compute() {

    double interaction_radius;
    ArrayList<int> nsides;
    Vector sidelengths;
    Vector mincoords;
    int nboxes, nterms;
    
    nsides.Init(rset_.n_rows());
    sidelengths.Init(rset_.n_rows());
    mincoords.Init(rset_.n_rows());

    printf("Computing FGT KDE...\n");
    fx_timer_start(NULL, "fgt_kde");

    // initialize densities to zero
    densities_.SetZero();

    FastGaussTransformPreprocess(&interaction_radius, nsides, sidelengths,
				 mincoords, &nboxes, &nterms);

    /*
      dyv_array *center = mk_dyv_array(nboxes, dim);
      dyv_array *locexp = mk_dyv_array_of_zeroed_dyvs(nboxes, 0);
      ivec_array *queries_assigned = mk_array_of_zero_length_ivecs(nboxes);
      ivec_array *references_assigned = mk_array_of_zero_length_ivecs(nboxes);
      dyv_array *mcoeffs = mk_dyv_array_of_zeroed_dyvs(nboxes, 0);
    
      gauss_t(q, d, e, delta, nterms, nboxes, nsides, sidelengths, mincoords,
      locexp, center, queries_assigned, references_assigned, mcoeffs);
    */

    fx_timer_stop(NULL, "fgt_kde");
  }

  void NormalizeDensities() {
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) *
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "fgt_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
