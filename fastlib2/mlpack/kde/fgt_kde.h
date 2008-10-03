/** @file fgt_kde.h
 *
 *  This file contains an implmentation of kernel density estimation
 *  using multidimensional version of the original fast Gauss
 *  transform for a linkable library component. This algorithm by
 *  design supports only the Gaussian kernel with the
 *  fixed-bandwidth. The optimal bandwidth cross-validation routine is
 *  not provided in this library.
 *
 *  For more details on mathematical derivations, please take a look
 *  at the following paper:
 *
 *  Article{ggstrain,
 *  Author = "L. Greengard and J. Strain", 
 *  Title = "{The Fast Gauss Transform}",
 *  Journal = "SIAM Journal of Scientific and Statistical Computing",
 *  Volume = "12(1)",
 *  Year = "1991",
 *  Pages = "79-94" }
 *
 *  TO-DO: Extend this code to nonuniform weights, and also replace
 *  all series expansion calls with those in series expansion library.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef FGT_KDE_H
#define FGT_KDE_H

#include <fastlib/fastlib.h>
#include "..//series_expansion/mult_series_expansion_aux.h"


/** @brief A computation class for FGT based kernel density estimation
 *
 *  This class is only inteded to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    FGTKde fgt_kde;
 *    struct datanode* fgt_kde_module;
 *    Vector results;
 *
 *    fgt_kde_module = fx_submodule(NULL, "kde", "fgt_kde_module");
 *    fgt_kde.Init(queries, references, fgt_kde_module);
 *    fgt_kde.Compute();
 *
 *    // important to make sure that you don't call Init on results!
 *    fgt_kde.get_density_estimates(&results);
 *  @endcode
 */
class FGTKde {
  
  FORBID_ACCIDENTAL_COPIES(FGTKde);

 private:

  ////////// Private Member Variables //////////

  /** @brief The datanode holding the parameters. */
  struct datanode *module_;

  /** @brief The column-oriented query dataset. */
  Matrix qset_;

  /** @brief The column-oriented reference dataset. */
  Matrix rset_;

  /** @brief The Gaussian kernel object. */
  GaussianKernel kernel_;

  /** @brief The vector holding the computed densiites. */
  Vector densities_;
  
  /** @brief Desired absolute error level. */
  double tau_;

  /** @brief Precomputed Taylor constants. */
  MultSeriesExpansionAux msea_;
  
  ////////// Private Member Functions //////////

  /** @brief Preprocessing for gridding the data points into boxes.
   *
   *  @param interaction_radius For each boxes that contain query points,
   *                            reference boxes whose distance from the
   *                            query boxes that are farther away than this
   *                            threshold will be ignored.
   *  @param nsides The number of grid boxes along each dimension.
   *  @param sidelengths The lengths of each hyperrectangle created in
   *                     gridding.
   *  @param mincoords The minimum coordinates of the bounding box
   *                   that encompasses the dataset in each dimension.
   *  @param nboxes The total number of boxes created.
   *  @param nterms The multivariate order of approximation is (nterms - 1).
   */
  void FastGaussTransformPreprocess_(double *interaction_radius, 
				     ArrayList<int> &nsides, 
				     Vector &sidelengths, Vector &mincoords, 
				     int *nboxes, int *nterms) {
  
    // Compute the interaction radius.
    double bandwidth = sqrt(kernel_.bandwidth_sq());
    *interaction_radius = sqrt(-2.0 * kernel_.bandwidth_sq() * log(tau_));

    int di, n, num_rows = rset_.n_cols();
    int dim = rset_.n_rows();

    // Discretize the grid space into boxes.
    Vector maxcoords;
    maxcoords.Init(dim);
    maxcoords.SetAll(-DBL_MAX);
    double boxside = -1.0;
    *nboxes = 1;

    for(di = 0; di < dim; di++) {
      mincoords[di] = DBL_MAX;
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

    // Figure out how many boxes lie along each direction.
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

    // Determine the truncation order.
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

  /** @brief Returns the index in a single-dim array, for the given
   *         coords in a d-dim array, with n[i] elements in the ith 
   *         dimension.
   *
   *  Basically, each grid box is labeled with a number in this
   *  fashion (2-D example follows):
   *
   *  y
   *  |
   *  |30 31 32 33 34 35 36 37 38 39
   *  |20 21 22 23 24 25 26 27 28 29
   *  |10 11 12 13 14 15 16 17 18 19
   *  | 0  1  2  3  4  5  6  7  8  9
   *  |_____________________________
   *                                 x
   *
   *  @param coords The coordinate of the point that we want to locate in the
   *                fast Gauss transform grid.
   *  @param n The number of boxes in each dimension. The i-th position of
   *           array tells how many boxes lie along the i-th dimension.
   */
  int MultiDimIndexInSingleArray_(ArrayList<int> &coords, 
				  ArrayList<int> &n) {
    
    int sum = 0;
    
    for(index_t i = 0; i < coords.size(); i++) {
      int prod = 1;
      for(index_t j = 0; j < i; j++) {
	prod *= n[j];
      }
      sum += coords[i] * prod;
    }
    return sum;
  }

  /** @brief Returns an ivec containing index translated into
   *  coordinates in {n[0], ..., n[d-1]} space.
   *
   *  @param n The i-th position of this array tells how many boxes lie
   *           along the i-th dimension.
   *  @param index The box number.
   *  @param coords The translated box coordinates.
   */
  void SingleDimIndexInMultiArray_(ArrayList<int> &n, int index, 
				   ArrayList<int> &coords) {

    for(index_t i = 0; i < coords.size(); i++) {
      int this_coord = index % n[i];
      index /= n[i];
      coords[i] = this_coord;
    }
  }

  /** @brief Determines whether (old_coords) + delta is on the grid box.
   *
   *  @param old_coords The original box coordinates.
   *  @param delta The perturbation to each box coordinate we would like to
   *               apply.
   *  @param new_coords The computed perturbed box coordinates.
   *  @param nsides The i-th position of this array tells how many boxes lie
   *                along the i-th dimension.
   *
   *  @return 1, if the box coordinate is within the grid, 0 otherwise.
   */
  int IsOngrid_(ArrayList<int> &old_coords, int delta, 
		ArrayList<int> &new_coords, ArrayList<int> &nsides) {
    
    int dim = old_coords.size();
    for(index_t d = 0; d < dim; d++) {
      int new_val = old_coords[d] + delta;
      
      if(new_val < 0 || new_val >= nsides[d]) {
	return 0;
      }
      
      new_coords[d] = new_val;
    }
    return 1;
  }

  /** @brief Compute the list of the neighboring boxes for a given box.
   *
   *  @param ibox The id of the grid box.
   *  @param nsides The i-th position tells how many boxes lie along the
   *                i-th dimension.
   *  @param kdis The number of neighbors to look for in increasing direction.
   *  @param ret The list of grid boxes that are considered neighbors for
   *             the grid box with id = ibox.
   */
  void MakeNeighbors_(int ibox, ArrayList<int> &nsides, int kdis, 
		      ArrayList<int> &ret) {

    // Compute actual vector position of a given box.
    ArrayList<int> coords;
    coords.Init(nsides.size());
    SingleDimIndexInMultiArray_(nsides, ibox, coords);

    ArrayList<int> dummy_n;
    dummy_n.Init(coords.size());
    ArrayList<int> all_ones;
    all_ones.Init(coords.size()); 
    for(index_t i = 0; i < coords.size(); i++) {
      dummy_n[i] = 2 * kdis + 1;
      all_ones[i] = kdis;
    }

    int num_neighbors = (2 * kdis + 1);
    int i;

    // number of neighbors in $D$ dimension is num_neighbors^D
    num_neighbors = (int) pow(num_neighbors, coords.size());

    // We need to generate every combination of [-1, 0, 1] (except [0, 0, 0]).
    // This is slightly hacky but also pretty crafty. We can easily enumerate 
    // these combinations by viewing the space as a grid in d dimensions with 
    // 3 cells in each. Therefore, we get coordinates in [0, 1, 2]. We achieve 
    // our goal by simply subtracting 1. We filter out values that aren't on 
    // the grid.
    ArrayList<int> delta, new_coords;
    delta.Init(coords.size());
    new_coords.Init(coords.size());

    for(i = 0; i < num_neighbors; i++) {
      SingleDimIndexInMultiArray_(dummy_n, i, delta);

      for(index_t j = 0; j < coords.size(); j++) {
	new_coords[j] = coords[j] + delta[j];
      }
      int ongrid = IsOngrid_(new_coords, -kdis, new_coords, nsides);
      
      if(ongrid) {
	*(ret.PushBackRaw()) = 
	  MultiDimIndexInSingleArray_(new_coords, nsides);
      }
      
    }
  }

  /** @brief Assigns all query and reference points to the grid.
   *
   *  @param nallbx The total number of grid boxes.
   *  @param nsides The number of grid boxes along each dimension.
   *  @param sidelengths The i-th position of this array tells the length of
   *                     the grid hypercube in the i-th dimension.
   *  @param mincoords The minimum coordinates of the fast Gauss transform
   *                   grid.
   *  @param center The center of each grid box.
   *  @param queries_assigned The i-th position of this array contains the
   *                          list of query indices assigned to the i-th
   *                          grid box.
   *  @param references_assigned The i-th position of this array contains the
   *                             list of reference indices assigned to the
   *                             i-th grid box.
   */
  void Assign_(int nallbx, ArrayList<int> &nsides, Vector &sidelengths, 
	       Vector &mincoords, Matrix &center, 
	       ArrayList<ArrayList<int> > &queries_assigned, 
	       ArrayList<ArrayList<int> > &references_assigned) {

    int r;
    int num_query_rows = qset_.n_cols();
    int num_ref_rows = rset_.n_cols();
    int dim = qset_.n_rows();
    int di;
  
    // Assign the reference points.
    for(r = 0; r < num_ref_rows; r++) {

      int boxnum = 0;

      for(di = dim - 1; di >= 0; di--) {
	int nside = nsides[di];
	double h = sidelengths[di];
	int binnum = (int) floor((rset_.get(di, r) - mincoords[di]) / h);
      
	binnum = max(0, min(binnum, nside - 1));
	boxnum = boxnum * nside + binnum;
      }
      *(references_assigned[boxnum].PushBackRaw()) = r;
    }

    // Assign the query points
    for(r = 0; r < num_query_rows; r++) {
    
      int boxnum = 0;

      for(di = dim - 1; di >= 0; di--) {
	int nside = nsides[di];
	double h = sidelengths[di];
	int binnum = (int) floor((qset_.get(di, r) - mincoords[di]) / h);
      
	binnum = max(0, min(binnum, nside - 1));
	boxnum = boxnum * nside + binnum;
      }
      *(queries_assigned[boxnum].PushBackRaw()) = r;
    }
  
    // Create centers for all boxes.
    for(r = 0; r < nallbx; r++) {
      int sf = nallbx;
      int ind = r, rem;
      Vector box_center;    
      center.MakeColumnVector(r, &box_center);

      for(di = dim - 1; di >= 0; di--) {
	int nside = nsides[di];
	double h = sidelengths[di];
	sf /= nside;
	rem = ind % sf;
	ind = ind / sf;

	box_center[di] = mincoords[di] + (ind + 0.5) * h;

	ind = rem;
      }
    }
  }

  /** @brief Translates a far-field expansion of a reference box and
   *         accumulates onto the local expansion of a given query box.
   *
   *  @param ref_box_num The box number of the references.
   *  @param query_box_num The box number of the queries.
   *  @param mcoeffsb The set of far-field moments. The i-th column of
   *                  this matrix contains the far-field moments for the
   *                  i-th box.
   *  @param lcoeffsb The set of local moments. The i-th column of this matrix
   *                  contains the local moments for the i-th box.
   *  @param p_alpha The approximation order is up to (p_alpha - 1).
   *  @param totalnumcoeffs The total number of coefficients.
   *  @param bwsqd_2 The squared bandwidth times two.
   *  @param hrcentroid The centroid of the the reference box.
   *  @param dest_hrcentroid The centroid the query box.
   */
  void TranslateMultipoleToLocal_(int ref_box_num, int query_box_num,
				  Matrix &mcoeffsb, Matrix &lcoeffsb,
				  int p_alpha, int totalnumcoeffs,
				  double bwsqd_2, const double *hrcentroid,
				  const double *dest_hrcentroid) {

    double bandwidth = sqrt(bwsqd_2);
    int j, k, l, d, step;

    int dim = qset_.n_rows();

    Vector lcoeffs;
    lcoeffsb.MakeColumnVector(query_box_num, &lcoeffs);
    Vector mcoeffs;
    mcoeffsb.MakeColumnVector(ref_box_num, &mcoeffs);
  
    {
      Vector dest_minus_parent;
      dest_minus_parent.Init(dim);
      
      la::SubOverwrite(dim, dest_hrcentroid, hrcentroid, 
		       dest_minus_parent.ptr());

      int limit = 2 * p_alpha - 2;
      Matrix hermite_map;
      hermite_map.Init(dim, limit + 1);
      Matrix arrtmp;
      arrtmp.Init(dim, totalnumcoeffs);

      Vector C_k_neg;
      C_k_neg.Alias(msea_.get_neg_inv_multiindex_factorials());
    
      for(j = 0; j < dim; j++) {
	double coord_div_band = dest_minus_parent[j] / bandwidth;
	double d2 = 2 * coord_div_band;
	double facj = exp(-coord_div_band * coord_div_band);
      
	hermite_map.set(j, 0, facj);
      
	if(p_alpha > 1) {
	  hermite_map.set(j, 1, d2 * facj);
	
	  for(k = 1; k < limit; k++) {
	    int k2 = k * 2;
	    hermite_map.set(j, k + 1,
			    d2 * hermite_map.get(j, k) - k2 * 
			    hermite_map.get(j, k - 1));
	  }
	}
	
	for(l = 0; l < totalnumcoeffs; l++) {
	  arrtmp.set(j, l, 0.0);
	}
      }
    
      step = totalnumcoeffs / p_alpha;
      d = 0;
    
      for(j = 0; j < totalnumcoeffs; j++) {
	const ArrayList<int> &mapping = msea_.get_multiindex(j);
	
	for(k = 0, l = j % step; k < p_alpha; k++, l += step) {
	  arrtmp.set(d, j, arrtmp.get(d, j) + mcoeffs[l] * 
		     hermite_map.get(d, mapping[d] + k));
	}
      }
    
    
      if(p_alpha > 1) {
	int boundary, boundary2;

	for(boundary = totalnumcoeffs / p_alpha, step = step / p_alpha, d = 1; 
	    step >= 1; step /= p_alpha, d++, boundary /= p_alpha) {
	
	  boundary2 = 0;
	
	  for(j = 0; j < totalnumcoeffs; j++) {
	    const ArrayList<int> &mapping = msea_.get_multiindex(j);

	    if(j % boundary == 0) {
	      boundary2 += boundary;
	    }
	  
	    for(k = 0; k < p_alpha; k++) {
	      
	      int jump = (j + step * k) % boundary2;
	    
	      if(jump < boundary2 - boundary) {
		jump += boundary2 - boundary;
	      }
	    
	      const ArrayList<int> &mapping2 = msea_.get_multiindex(jump);

	      arrtmp.set(d, j,
			 arrtmp.get(d, j) +
			 arrtmp.get(d - 1, jump) * 
			 hermite_map.get(d, mapping2[d] + mapping[d]));
	    }
	  }
	}
      }
    
      d = dim - 1;
      
      for(j = 0; j < totalnumcoeffs; j++) {
	lcoeffs[j] = lcoeffs[j] + C_k_neg[j] * arrtmp.get(d, j);
      }
    }
  }

  /** @brief Compute far-field moments for a given reference box.
   *
   *  @param mcoeffs The i-th column of this matrix contains the far-field
   *                 moments contributed by the reference points contained
   *                 within.
   *  @param dim The dimensionality.
   *  @param p_alpha The approximation order.
   *  @param totalnumcoeffs The total number of coefficients.
   *  @param ref_box_num The id of the grid box containing the reference
   *                     points.
   *  @param rows The ids of the reference points contained within the box.
   *  @param x_R The center of the reference box.
   */
  void ComputeMultipoleCoeffs_(Matrix &mcoeffs, int dim,
			       int p_alpha, int totalnumcoeffs, 
			       int ref_box_num, double bwsqd_two, 
			       ArrayList<int> &rows, const double *x_R) {
    
    Vector A_k;
    mcoeffs.MakeColumnVector(ref_box_num, &A_k);
    double bw_times_sqrt_two = sqrt(bwsqd_two);

    // If the thing has been computed already, return. Otherwise, compute it
    // and store it as a cached sufficient statistics.
    if(A_k[0] != 0) {
      return;
    }

    Vector C_k;
    C_k.Alias(msea_.get_inv_multiindex_factorials());
  
    Vector tmp;
    tmp.Init(totalnumcoeffs);
    Vector x_r;
    x_r.Init(dim);
    int num_rows = rows.size();
    int step, boundary;
    int r, i, j;
  
    A_k[0] = num_rows;
    for(i = 1; i < totalnumcoeffs; i++) {
      A_k[i] = 0.0;
    }
  
    if(p_alpha > 1) {
    
      for(r = 0; r < num_rows; r++) {
      
	int row_num = rows[r];
      
	for(i = 0; i < dim; i++) {
	  x_r[i] = (rset_.get(i, row_num) - x_R[i]) / bw_times_sqrt_two;
	}
      
	tmp[0] = 1.0;
      
	for(boundary = totalnumcoeffs, step = totalnumcoeffs / p_alpha,
	      j = 0;
	    step >= 1; step /= p_alpha, boundary /= p_alpha, j++) {
	  for(i = 0; i < totalnumcoeffs; ) {
	    int limit = i + boundary;
	  
	    i += step;
	  
	    for( ; i < limit; i += step) {
	      tmp[i] = tmp[i - step] * x_r[j];
	    }
	  }
	}
      
      
	for(i = 1; i < totalnumcoeffs; i++) {
	  A_k[i] = A_k[i] + tmp[i];
	}
      }
    }
  
    for(r = 1; r < totalnumcoeffs; r++) {
      A_k[r] = A_k[r] * C_k[r];
    }

    return;
  }

  /** @brief Directly accumulate the contribution of a given reference box
   *         into the local moments for a given query box.
   *
   *  @param rows The ids of the reference points contained within the
   *              reference box.
   *  @param query_box_num The box number of the query box.
   *  @param locexps The i-th column of this matrix contains the local
   *                 moments of the i-th grid box.
   *  @param delta Twice the squared bandwidth.
   *  @param dest_hrcentroid The center of the query box.
   *  @param p_alpha The approximation order.
   *  @param totalnumcoeffs The total number of coefficients.
   */
  void DirectLocalAccumulation_(ArrayList<int> &rows, int query_box_num,
				Matrix &locexps, double delta,
				const double *dest_hrcentroid, int p_alpha, 
				int totalnumcoeffs) {

    int num_rows = rows.size();
    int r, d, boundary, step, i, j, k;

    // Retrieve the centroid
    int dim = rset_.n_rows();
    int limit2 = p_alpha - 1;
    Matrix hermite_map;
    hermite_map.Init(dim, p_alpha);
    Vector arrtmp;
    arrtmp.Init(totalnumcoeffs);
    Vector x_r_minus_x_Q;
    x_r_minus_x_Q.Init(dim);
    double bandwidth = sqrt(delta);
    Vector neg_inv_multiindex_factorials;
    neg_inv_multiindex_factorials.Alias
      (msea_.get_neg_inv_multiindex_factorials());

    Vector arr;
    locexps.MakeColumnVector(query_box_num, &arr);

    // For each reference point.
    for(r = 0; r < num_rows; r++) {

      int row_num = rows[r];
    
      // Calculate (x_r - x_Q)
      for(d = 0; d < dim; d++) {
	x_r_minus_x_Q[d] = dest_hrcentroid[d] - rset_.get(d, row_num);
      }
    
      // Compute the necessary Hermite precomputed map based on the
      // coordinate diference.
      for(d = 0; d < dim; d++) {

	double coord_div_band = x_r_minus_x_Q[d] / bandwidth;
	double d2 = 2 * coord_div_band;
	double facj = exp(-coord_div_band * coord_div_band);
      
	hermite_map.set(d, 0, facj);
      
	if(p_alpha > 1) {
	  hermite_map.set(d, 1, d2 * facj);
	
	  for(k = 1; k < limit2; k++) {
	    int k2 = k * 2;
	    hermite_map.set(d, k + 1,
			    d2 * hermite_map.get(d, k) - k2 * 
			    hermite_map.get(d, k - 1));
	  }
	}
      }
    
      // Seed to start out the coefficients.
      arrtmp[0] = 1.0;
    
      if(p_alpha > 1) {
      
	// Compute the Taylor coefficients directly...
	for(boundary = totalnumcoeffs, step = totalnumcoeffs / p_alpha,
	      d = 0;
	    step >= 1; step /= p_alpha, boundary /= p_alpha, d++) {
	  for(i = 0; i < totalnumcoeffs; ) {
	    int limit = i + boundary;
	  
	    // Skip the first one.
	    int first = i;
	    i += step;
	  
	    for(j = 1; i < limit; i += step, j++) {
	      arrtmp[i] = arrtmp[first] * hermite_map.get(d, j);
	    }
	  
	    arrtmp[first] *= hermite_map.get(d, 0);
	  }
	}
      }
      else {
	for(d = 0; d < dim; d++) {
	  arrtmp[0] *= hermite_map.get(d, 0);
	}
      }
    
      for(j = 0; j < totalnumcoeffs; j++) {
	arr[j] = arr[j] + neg_inv_multiindex_factorials[j] * arrtmp[j];
      }    
    }
  }

  /** @brief Evaluate far-field expansion of a reference box for a set of
   *         query points.
   *
   *  @param rows The set of query points for which we want to evaluate
   *              the far-field expansion.
   *  @param p_alpha The order of approximation is (p_alpha - 1).
   *  @param totalnumcoeffs The total number of coefficients.
   *  @param mcoeffsb The set of far-field moments. The i-th column of
   *                  this matrix contains the far-field moments for the
   *                  i-th box.
   *  @param ref_box_num The box number of the references.
   *  @param bwsqd_times_2 The squared bandwidth times two.
   *  @param ref_hrcentroid The center of the reference box.
   */
  void EvaluateMultipoleExpansion_(ArrayList<int> &rows, int p_alpha,
				   int totalnumcoeffs, Matrix &mcoeffsb,
				   int ref_box_num, double bwsqd_times_2, 
				   const double *ref_hrcentroid) {
    
    double bandwidth = sqrt(bwsqd_times_2);
    int num_query_rows = rows.size();
    int r, d, boundary, step, i, j, k;

    // Retrieve the reference centroid.
    int dim = qset_.n_rows();
    Vector x_q_minus_x_R;
    x_q_minus_x_R.Init(dim);
    Matrix hermite_map;
    hermite_map.Init(dim, p_alpha);
    Vector arrtmp;
    arrtmp.Init(totalnumcoeffs);
    int limit2 = p_alpha - 1;
    Vector mcoeffs;
    mcoeffsb.MakeColumnVector(ref_box_num, &mcoeffs);
  
    // For each data point,
    for(r = 0; r < num_query_rows; r++) {

      double multipolesum = 0.0;
      int row_num = rows[r];

      // Calculate (x_q - x_R)
      for(d = 0; d < dim; d++) {
	x_q_minus_x_R[d] = qset_.get(d, row_num) - ref_hrcentroid[d];
      }
      
      // Compute the necessary Hermite precomputed map based on the coordinate
      // difference.
      for(d = 0; d < dim; d++) {
	double coord_div_band = x_q_minus_x_R[d] / bandwidth;
	double d2 = 2 * coord_div_band;
	double facj = exp(-coord_div_band * coord_div_band);
      
	hermite_map.set(d, 0, facj);
      
	if(p_alpha > 1) {
	  hermite_map.set(d, 1, d2 * facj);
	
	  for(k = 1; k < limit2; k++) {
	    int k2 = k * 2;
	    hermite_map.set(d, k + 1,
			    d2 * hermite_map.get(d, k) - k2 * 
			    hermite_map.get(d, k - 1));
	  }
	}
      }
    
      // Seed to start out the coefficients.
      arrtmp[0] = 1.0;
    
      if(p_alpha > 1) {
      
	for(boundary = totalnumcoeffs, step = totalnumcoeffs / p_alpha,
	      d = 0;
	    step >= 1; step /= p_alpha, boundary /= p_alpha, d++) {
	  for(i = 0; i < totalnumcoeffs; ) {
	    int limit = i + boundary;
	  
	    // Skip the first one.
	    int first = i;
	    i += step;
	    
	    for(j = 1; i < limit; i += step, j++) {
	      arrtmp[i] = arrtmp[first] * hermite_map.get(d, j);
	    }
	  
	    arrtmp[first] *= hermite_map.get(d, 0);
	  }
	}
      }
      else {
	for(d = 0; d < dim; d++) {
	  arrtmp[0] *= hermite_map.get(d, 0);
	}
      }
    
      for(j = 0; j < totalnumcoeffs; j++) {
	multipolesum += mcoeffs[j] * arrtmp[j];
      }

      densities_[row_num] += multipolesum;
    }
  }

  /** @brief Evaluate local expansion for a single query point.
   *
   *  @param row_q The id of the query point.
   *  @param x_Q The centroid of the query box containing the the query point.
   *  @param h The bandwidth.
   *  @param query_box_num The box number of the query box.
   *  @param lcoeffsb The i-th column of this matrix contains the local
   *                  moments accumulated for the i-th grid box.
   *  @param totalnumcoeffs The total number of coefficients.
   *  @param p_alpha The approximation order.
   *
   *  @return The evaluated local expansion value.
   */
  double EvaluateLocalExpansion_(int row_q, Vector &x_Q, double h, 
				 int query_box_num, Matrix &lcoeffsb, 
				 int totalnumcoeffs, int p_alpha) {
    
    int dim = qset_.n_rows();
    Vector x_Q_to_x_q;
    x_Q_to_x_q.Init(dim);
    int i, j, boundary, step;
    double multipolesum = 0;

    /**
     * First calculate (x_q - x_Q) / (sqrt(2) * h)
     */
    for(i = 0; i < dim; i++) {
      double x_Q_val = x_Q[i];
      double x_q_val = qset_.get(i, row_q);
      x_Q_to_x_q[i] = (x_q_val - x_Q_val) / h;
    }

    {
      Vector lcoeffs;
      lcoeffsb.MakeColumnVector(query_box_num, &lcoeffs);

      Vector tmp;
      tmp.Init(totalnumcoeffs);
      tmp[0] = 1.0;
    
      if(totalnumcoeffs > 1) {
	
	for(boundary = totalnumcoeffs, step = totalnumcoeffs / p_alpha,
	      j = 0;
	    step >= 1; step /= p_alpha, boundary /= p_alpha, j++) {
	  for(i = 0; i < totalnumcoeffs; ) {
	    int limit = i + boundary;
	  
	    i += step;
	  
	    for( ; i < limit; i += step) {
	      tmp[i] = tmp[i - step] * x_Q_to_x_q[j];
	    }
	  }
	}    
      }
    
      for(i = 0; i < totalnumcoeffs; i++) {
	multipolesum += lcoeffs[i] * tmp[i];
      }
    }
    return multipolesum;
  }

  /** @brief Go through each query boxes and evaluate the local
   *         expansions accumulated in each box.
   *
   *  @param delta Twice the squared bandwidth.
   *  @param nterms The approximation order.
   *  @param nallbx The total number of grid boxes.
   *  @param nsides The number of grid boxes along each dimension.
   *  @param locexp The i-th column of this matrix contains the local moments
   *                for the i-th grid box.
   *  @param nlmax  For a given pair of a reference box with computed far-field
   *                moments and a query box, if the number of query points
   *                is within this limit, then we evaluate the contribution
   *                of the reference box by direct far-field evaluations.
   *                Otherwise, the far-field moments are converted into
   *                local moments.
   *  @param queries_assigned The i-th column of this matrix contains the
   *                          ids of the queries assigned to the i-th grid
   *                          box.
   *  @param center The i-th column of this matrix contains the coordinates
   *                of the i-th grid box.
   *  @param totalnumcoeffs The total number of coefficients.
   */
  void EvaluateLocalExpansionForAllQueries_
    (double delta, int nterms, int nallbx, ArrayList<int> &nsides, 
     Matrix &locexp, int nlmax, ArrayList<ArrayList<int> > &queries_assigned, 
     Matrix &center, int totalnumcoeffs) {

    int i, j;
    
    // GO through all query boxes.
    for(i = 0; i < nallbx; i++) {
      ArrayList<int> &query_rows = queries_assigned[i];
      int ninbox = query_rows.size();
      
      if(ninbox <= nlmax) {
	continue;
      }
      else {
	
	for(j = 0; j < ninbox; j++) {
	  int row_q = query_rows[j];
	  Vector x_Q;
	  center.MakeColumnVector(i, &x_Q);
	  
	  double result = EvaluateLocalExpansion_(row_q, x_Q, sqrt(delta), 
						  i, locexp, totalnumcoeffs, 
						  nterms);
	  densities_[row_q] += result;
	}
      }
    }
  }

  /** @brief The main workhorse of the algorithm that does direct evaluation,
   *         far-field approximation, direct local accumulation, and
   *         far-field-to local translations, i.e. the FGT algorithm.
   *
   *  @param delta
   *  @param nterms
   *  @param nallbx The total number of grid boxes in the FGT grid.
   *  @param mincoords The minimum coordinates of the FGT grid.
   *  @param locexp The i-th column of this matrix contains the local moments
   *                accumulated for the i-th grid box.
   *  @param nfmax If the number of reference points owned by a given grid box
   *               is within this limit, then no far-field moments are
   *               computed.
   *  @param nlmax For a given pair of a reference box with computed far-field
   *               moments and a query box, if the number of query points
   *               is within this limit, then we evaluate the contribution
   *               of the reference box by direct far-field evaluations.
   *               Otherwise, the far-field moments are converted into
   *               local moments.
   *  @param kdis
   *  @param center The i-th column of this matrix contains the coordinates
   *                of the center of each grid box.
   *  @param queries_assigned The i-th column of this matrix contains the
   *                          ids of the queries assigned to the i-th
   *                          grid box.
   *  @param references_assigned The i-th column of this matrix contains the
   *                             ids of the reference points assigned to the
   *                             i-th grid box.
   *  @param mcoeffs The i-th column of this matrix contains the far-field
   *                 moments of the i-th box contributed by the reference
   *                 points contained within.
   */
  void FinalizeSum_(double delta, int nterms, int nallbx, 
		    ArrayList<int> &nsides, Vector &sidelengths, 
		    Vector &mincoords, Matrix &locexp, int nfmax, int nlmax, 
		    int kdis, Matrix &center, 
		    ArrayList<ArrayList<int> > &queries_assigned,
		    ArrayList<ArrayList<int> > &references_assigned, 
		    Matrix &mcoeffs) {

    int dim = qset_.n_rows();

    int totalnumcoeffs = (int) pow(nterms, dim);

    // Step 1: Assign query points and reference points to boxes.
    Assign_(nallbx, nsides, sidelengths, mincoords, center, 
	    queries_assigned, references_assigned);

    // Initialize local expansions to zero.
    int i, j, k, l;

    // Process all reference boxes
    for(i = 0; i < nallbx; i++) {

      ArrayList<int> &reference_rows = references_assigned[i];
      int ninbox = reference_rows.size();

      // If the box contains no reference points, skip it.
      if(ninbox <= 0) {
	continue;
      }

      // In this case, no far-field expansion is created
      else if(ninbox <= nfmax) {

	// Get the query boxes that are in the interaction range.
	ArrayList <int> nbors;
	nbors.Init();
	MakeNeighbors_(i, nsides, kdis, nbors);
	int nnbors = nbors.size();

	for(j = 0; j < nnbors; j++) {

	  // Number of query points in this neighboring box.
	  int query_box_num = nbors[j];
	  ArrayList<int> &query_rows = queries_assigned[query_box_num];
	  int ninnbr = query_rows.size();
	
	  if(ninnbr <= nlmax) {

	    // Direct Interaction
	    for(k = 0; k < ninnbr; k++) {
	    
	      int query_row = query_rows[k];
	      const double *query = qset_.GetColumnPtr(query_row);

	      for(l = 0; l < ninbox; l++) {
		int reference_row = reference_rows[l];
		const double *reference = rset_.GetColumnPtr(reference_row);

		double dsqd = 
		  la::DistanceSqEuclidean(qset_.n_rows(), query, reference);

		double pot = exp(-dsqd / delta);

		// Here, I hard-coded to do only one bandwidth.
		densities_[query_row] += pot;
	      }
	    }
	  
	  }

	  // In this case, take each reference point and convert into the 
	  // Taylor series.
	  else {
	  
	    DirectLocalAccumulation_(reference_rows, query_box_num,
				     locexp, delta,
				     center.GetColumnPtr(query_box_num),
				     nterms, totalnumcoeffs);
	  }
	
	}
      }

      // In this case, create a far field expansion.
      else {

	ComputeMultipoleCoeffs_(mcoeffs ,dim, nterms,
				totalnumcoeffs, i, delta, reference_rows,
				center.GetColumnPtr(i));

	// Get the query boxes that are in the interaction range.
	ArrayList<int> nbors;
	nbors.Init();
	MakeNeighbors_(i, nsides, kdis, nbors);
	int nnbors = nbors.size();

	for(j = 0; j < nnbors; j++) {
	  int query_box_num = nbors[j];
	  ArrayList<int> &query_rows = queries_assigned[query_box_num];
	  int ninnbr = query_rows.size();

	  // If this is true, evaluate far field expansion at each query point.
	  if(ninnbr <= nlmax) {
	    EvaluateMultipoleExpansion_(query_rows, nterms,
					totalnumcoeffs, mcoeffs,
					i, delta, center.GetColumnPtr(i));
	  }

	  // In this case do far-field to local translation.
	  else {
	    
	    TranslateMultipoleToLocal_(i, query_box_num,
				       mcoeffs, locexp,
				       nterms, totalnumcoeffs, delta,
				       center.GetColumnPtr(i),
				       center.GetColumnPtr(query_box_num));
	  
	  }
	}
      }
    }

    // Now evaluate the local expansions for all queries.
    EvaluateLocalExpansionForAllQueries_(delta, nterms, nallbx, nsides, locexp,
					 nlmax, queries_assigned, center, 
					 totalnumcoeffs);
  }

  /** @brief Determines the cut-off ranges for efficient evaluations and
   *         calls the main workhorse for the algorithm.
   *
   *  @param delta The twice the squared bandwidth value.
   *  @param nterms The truncation order of the approximation.
   *  @param nallbx The total number of boxes.
   *  @param nsides The number of grid boxes along each dimension.
   *  @param sidelengths The length of the side of each grid box in each
   *                     dimensin.
   *  @param mincoords The minimum coordinates of the FGT grid.
   *  @param locexp The i-th column of this matrix contains the local moments
   *                accumulated for the i-th grid box.
   *  @param center The i-th column of this matrix contains the coordinates
   *                of the center of the i-th grid box.
   *  @param queries_assigned The i-th column of this matrix contains the
   *                          ids of the query points assigned to the i-th
   *                          grid box.
   *  @param references_assigned The i-th column of this matrix contains the
   *                             ids of the reference points assigned to the
   *                             i-th grid box.
   *  @param mcoeffs The i-th column of this matrix contains the far-field
   *                 moments for the i-the grid box contributed by the
   *                 reference points contained within.
   */
  void GaussTransform_(double delta, int nterms, int nallbx, 
		       ArrayList<int> &nsides, 
		       Vector &sidelengths, Vector &mincoords,
		       Matrix &locexp, Matrix &center,
		       ArrayList<ArrayList<int> > &queries_assigned, 
		       ArrayList<ArrayList<int> > &references_assigned,
		       Matrix &mcoeffs) {

    int dim = qset_.n_rows();
    int kdis = (int) (sqrt(log(tau_) * -2.0) + 1);

    // This is a slight modification of Strain's cutoff since he never
    // implemented this above 2 dimensions.
    int nfmax = (int) pow(nterms, dim - 1) + 2;
    int nlmax = nfmax;

    // Call gafexp to create all expansions on grid, evaluate all appropriate
    // far-field expansions and evaluate all appropriate direct interactions.
    FinalizeSum_(delta, nterms, nallbx, nsides, sidelengths, mincoords,
		 locexp, nfmax, nlmax, kdis, center, queries_assigned,
		 references_assigned, mcoeffs);
  }

  /** @brief Normalize the density estimates after the unnormalized
   *         sums have been computed.
   */
  void NormalizeDensities_() {
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) *
      rset_.n_cols();

    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
  }

 public:

  ////////// Constructor/Destructor //////////
  
  /** @brief Constructor that does not do anything. */
  FGTKde() {}
  
  /** @brief Destructor that does not do anything. */
  ~FGTKde() {}
  
  ////////// Getters/Setters //////////

  /** @brief Get the density estimates.
   *
   *  @param results An uninitialized vector which will be initialized
   *                 with the computed density estimates.
   */
  void get_density_estimates(Vector *results) {
    results->Init(densities_.length());

    for(index_t i = 0; i < densities_.length(); i++) {
      (*results)[i] = densities_[i];
    }
  }

  ///////// Initialization and computation //////////

  /** @brief Initialize with the given query and the reference
   *         datasets.
   *
   *  @param qset The column-oriented query dataset.
   *  @param rset The column-oriented reference dataset.
   *  @param module_in The module holding the parameters for execution.
   */
  void Init(Matrix &qset, Matrix &rset, struct datanode *module_in) {

    // initialize with the incoming module holding the paramters
    module_ = module_in;

    // initialize the kernel
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // set aliases to the query and reference datasets and initialize
    // query density sets
    qset_.Copy(qset);
    densities_.Init(qset_.n_cols());
    rset_.Copy(rset);

    // set accuracy
    tau_ = fx_param_double(module_, "absolute_error", 0.1);
  }

  /** @brief Compute KDE estimates using fast Gauss transform.
   */
  void Compute() {

    double interaction_radius;
    ArrayList<int> nsides;
    Vector sidelengths;
    Vector mincoords;
    int nboxes, nterms;
    int dim = rset_.n_rows();
    
    nsides.Init(rset_.n_rows());
    sidelengths.Init(rset_.n_rows());
    mincoords.Init(rset_.n_rows());

    printf("Computing FGT KDE...\n");
    
    // initialize densities to zero
    densities_.SetZero();

    fx_timer_start(module_, "fgt_kde_init");
    FastGaussTransformPreprocess_(&interaction_radius, nsides, sidelengths,
				  mincoords, &nboxes, &nterms);
    fx_timer_stop(module_, "fgt_kde_init");

    // precompute factorials
    msea_.Init(nterms - 1, qset_.n_rows());
    
    // stores the coordinate of each grid box
    Matrix center;
    center.Init(dim, nboxes);

    // stores the local expansion of each grid box
    Matrix locexp;
    locexp.Init((int) pow(nterms, dim), nboxes);
    locexp.SetZero();

    // stores the ids of query points assigned to each grid box
    ArrayList<ArrayList<int> > queries_assigned;
    queries_assigned.Init(nboxes);
    
    for(index_t i = 0; i < nboxes; i++) {
      queries_assigned[i].Init();
    }

    // stores the ids of reference points assigned to each grid box
    ArrayList<ArrayList<int> > references_assigned;
    references_assigned.Init(nboxes);

    for(index_t i = 0; i < nboxes; i++) {
      references_assigned[i].Init();
    }

    // stores the multipole moments of the reference points in each grid box
    Matrix mcoeffs;
    mcoeffs.Init((int)pow(nterms, dim), nboxes);
    mcoeffs.SetZero();
    
    double delta = 2 * kernel_.bandwidth_sq();

    fx_timer_start(module_, "fgt_kde");
    GaussTransform_(delta, nterms, nboxes, nsides, sidelengths, mincoords,
		    locexp, center, queries_assigned, references_assigned, 
		    mcoeffs);

    // normalize the sum
    NormalizeDensities_();
    fx_timer_stop(module_, "fgt_kde");
    printf("FGT KDE completed...\n");
  }

  /** @brief Output KDE results to a stream 
   *
   *  If the user provided "--fgt_kde_output=" argument, then the
   *  output will be directed to a file whose name is provided after
   *  the equality sign.  Otherwise, it will be provided to the
   *  screen.
   */
  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(module_, "fgt_kde_output", NULL)) != NULL) {
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
