/** @file fgt_kde.h
 *
 *  This file contains an implmentation of kernel density estimation
 *  using multidimensional version of the original fast Gauss
 *  transform for a linkable library component. This algorithm by
 *  design supports only the Gaussian kernel with the
 *  fixed-bandwidth. The optimal bandwidth cross-validation routine is
 *  not provided in this library.
 *
 *  For more details on nmathematical derivations, please take a look
 *  at the following paper:
 *
 * @Article{ggstrain,
 *  Author = "L. Greengard and J. Strain", 
 *  Title = "{The Fast Gauss Transform}",
 *  Journal = "SIAM Journal of Scientific and Statistical Computing",
 *  Volume = "12(1)",
 *  Year = "1991",
 *  Pages = "79-94" }
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bugs No known bugs.
 */

#ifndef FGT_KDE_H
#define FGT_KDE_H

#include <fastlib/fastlib.h>
#include "u/dongryel/series_expansion/mult_series_expansion_aux.h"


/**
 * A computation class for FGT based kernel density estimation
 *
 * This class is only inteded to compute once per instantiation.
 *
 * Example use:
 *
 * @code
 *   FGTKde fgt_kde;
 *   struct datanode* fgt_kde_module;
 *   Vector results;
 *
 *   fgt_kde_module = fx_submodule(NULL, "kde", "fgt_kde_module");
 *   fgt_kde.Init(queries, references, fgt_kde_module);
 *   fgt_kde.Compute();
 *
 *   // important to make sure that you don't call Init on results!
 *   fgt_kde.get_density_estimates(&results);
 * @endcode
 */
class FGTKde {
  
  FORBID_ACCIDENTAL_COPIES(FGTKde);

 private:

  ////////// Private Member Variables //////////

  /** datanode holding the parameters */
  struct datanode *module_;

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

  /** precomputed constants */
  MultSeriesExpansionAux msea_;

  /* returns the index in a single-dim array, for the given coords in a
   * d-dim array, with n[i] elements in the ith dimension
   */
  int multi_dim_index_in_single_array(ArrayList<int> &coords, 
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

  /** Returns an ivec containing index translated into coordinates in
   * {n[0],  ..., n[d-1]} space.
   */
  void single_dim_index_in_multi_array(ArrayList<int> &n, int index, 
				       ArrayList<int> &coords) {

    for(index_t i = 0; i < coords.size(); i++) {
      int this_coord = index % n[i];
      index /= n[i];
      coords[i] = this_coord;
    }
  }

  int is_ongrid(ArrayList<int> &old_coords, int delta, 
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

  void mknbor(int ibox, ArrayList<int> &nsides, int kdis, 
	      ArrayList<int> &ret) {

    // Compute actual vector position of a given box.
    ArrayList<int> coords;
    coords.Init(nsides.size());
    single_dim_index_in_multi_array(nsides, ibox, coords);

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
      single_dim_index_in_multi_array(dummy_n, i, delta);

      for(index_t j = 0; j < coords.size(); j++) {
	new_coords[j] = coords[j] + delta[j];
      }
      int ongrid = is_ongrid(new_coords, -kdis, new_coords, nsides);
      
      if(ongrid) {
	*(ret.AddBack()) = 
	  multi_dim_index_in_single_array(new_coords, nsides);
      }
      
    }
  }

  void assign(int nallbx, ArrayList<int> &nsides, Vector &sidelengths, 
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
      *(references_assigned[boxnum].AddBack()) = r;
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
      *(queries_assigned[boxnum].AddBack()) = r;
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

  void TranslateMultipoleToLocal(int ref_box_num, int query_box_num,
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

  void ComputeMultipoleCoeffs(Matrix &mcoeffs, int dim,
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


  void DirectLocalAccumulation(ArrayList<int> &rows, int query_box_num,
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

    /**
     * For each data point,
     */
    for(r = 0; r < num_rows; r++) {

      int row_num = rows[r];
    
      /**
       * Calculate (x_r - x_Q)
       */
      for(d = 0; d < dim; d++) {
	x_r_minus_x_Q[d] = dest_hrcentroid[d] - rset_.get(d, row_num);
      }
    
      /**
       * Compute the necessary Hermite precomputed map based on the coordinate
       * difference.
       */
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
    
      /**
       * Seed to start out the coefficients.
       */
      arrtmp[0] = 1.0;
    
      if(p_alpha > 1) {
      
	/**
	 * Compute the Taylor coefficients directly...
	 */
	for(boundary = totalnumcoeffs, step = totalnumcoeffs / p_alpha,
	      d = 0;
	    step >= 1; step /= p_alpha, boundary /= p_alpha, d++) {
	  for(i = 0; i < totalnumcoeffs; ) {
	    int limit = i + boundary;
	  
	    /**
	     * Skip the first one.
	     */
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

  void EvaluateMultipoleExpansion(ArrayList<int> &rows, int p_alpha,
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

  double compute_v_alpha(int row_q, Vector &x_Q, double h, 
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

  /**
   * Go through each query boxes and evaluate the local expansions
   * accumulated in each box.
   */
  void gaeval(double delta, int nterms, int nallbx,
	      ArrayList<int> &nsides, Matrix &locexp, int nlmax,
	      ArrayList<ArrayList<int> > &queries_assigned, 
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
	  
	  double result = compute_v_alpha(row_q, x_Q, sqrt(delta), 
					  i, locexp, totalnumcoeffs, nterms);
	  densities_[row_q] += result;
	}
      }
    }
  }

  void gafexp(double delta, int nterms,
	      int nallbx, ArrayList<int> &nsides, Vector &sidelengths, 
	      Vector &mincoords, Matrix &locexp, int nfmax, int nlmax, 
	      int kdis, Matrix &center, 
	      ArrayList<ArrayList<int> > &queries_assigned,
	      ArrayList<ArrayList<int> > &references_assigned, 
	      Matrix &mcoeffs) {

    int dim = qset_.n_rows();

    int totalnumcoeffs = (int) pow(nterms, dim);

    // Step 1: Assign query points and reference points to boxes.
    assign(nallbx, nsides, sidelengths, mincoords, center, 
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
	mknbor(i, nsides, kdis, nbors);
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
	  
	    DirectLocalAccumulation(reference_rows, query_box_num,
				    locexp, delta,
				    center.GetColumnPtr(query_box_num),
				    nterms, totalnumcoeffs);
	  }
	
	}
      }

      // In this case, create a far field expansion.
      else {

	ComputeMultipoleCoeffs(mcoeffs ,dim, nterms,
			       totalnumcoeffs, i, delta, reference_rows,
			       center.GetColumnPtr(i));

	// Get the query boxes that are in the interaction range.
	ArrayList<int> nbors;
	nbors.Init();
	mknbor(i, nsides, kdis, nbors);
	int nnbors = nbors.size();

	for(j = 0; j < nnbors; j++) {
	  int query_box_num = nbors[j];
	  ArrayList<int> &query_rows = queries_assigned[query_box_num];
	  int ninnbr = query_rows.size();

	  // If this is true, evaluate far field expansion at each query point.
	  if(ninnbr <= nlmax) {
	    EvaluateMultipoleExpansion(query_rows, nterms,
				       totalnumcoeffs, mcoeffs,
				       i, delta, center.GetColumnPtr(i));
	  }

	  // In this case do multipole to local translation.
	  else {
	    
	    TranslateMultipoleToLocal(i, query_box_num,
				      mcoeffs, locexp,
				      nterms, totalnumcoeffs, delta,
				      center.GetColumnPtr(i),
				      center.GetColumnPtr(query_box_num));
	  
	  }
	}
      }
    }

    gaeval(delta, nterms, nallbx, nsides, locexp, nlmax,
	   queries_assigned, center, totalnumcoeffs);

  }

  void gauss_t(double delta, int nterms, int nallbx, ArrayList<int> &nsides, 
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
    gafexp(delta, nterms, nallbx, nsides, sidelengths, mincoords,
	   locexp, nfmax, nlmax, kdis, center, queries_assigned,
	   references_assigned, mcoeffs);
  }

 public:

  ////////// Constructor/Destructor //////////
  
  /** constructor */
  FGTKde() {}
  
  /** destructor */
  ~FGTKde() {}
  
  ////////// Getters/Setters //////////

  /** get the density estimate */
  void get_density_estimates(Vector *results) {
    results->Init(densities_.length());

    for(index_t i = 0; i < densities_.length(); i++) {
      (*results)[i] = densities_[i];
    }
  }

  ///////// Initialization and computation //////////

  /** initialize with the given query and the reference datasets */
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

  void FastGaussTransformPreprocess(double *interaction_radius, 
				    ArrayList<int> &nsides, 
				    Vector &sidelengths, Vector &mincoords, 
				    int *nboxes, int *nterms) {
  
    // Compute the interaction radius.
    double bandwidth = sqrt(kernel_.bandwidth_sq());
    *interaction_radius = sqrt(-2.0 * kernel_.bandwidth_sq() * log(tau_));

    int di, n, num_rows = rset_.n_cols();
    int dim = rset_.n_rows();

    /**
     * Discretize the grid space into boxes.
     */
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

  /** 
   * Compute KDE estimates using fast Gauss transform.
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
    FastGaussTransformPreprocess(&interaction_radius, nsides, sidelengths,
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
    gauss_t(delta, nterms, nboxes, nsides, sidelengths, mincoords,
	    locexp, center, queries_assigned, references_assigned, mcoeffs);

    // normalize the sum
    NormalizeDensities();
    fx_timer_stop(module_, "fgt_kde");
    printf("FGT KDE completed...\n");
  }

  /** 
   * Normalize the density estimates after the unnormalized sums have
   * been computed 
   */
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
