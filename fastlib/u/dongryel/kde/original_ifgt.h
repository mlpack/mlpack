/** @file original_ifgt.h
 *
 *  Revision: 0.99
 *  Date	: Mon Jul  8 18:11:31 EDT 2002
 *
 *  Revision: 1.00
 *  Date : Wed Jan 29 16:05:21 EDT 2003
 *
 *  Revision: 1.01
 *  Date : Wed Apr 14 09:51:26 EDT 2004
 *
 *  This implements the KDE using the original improved fast Gauss
 *  transform algorithm. For more details, take a look at:
 *
 *  inproceedings{946593, author = {Changjiang Yang and Ramani
 *  Duraiswami and Nail A. Gumerov and Larry Davis}, title = {Improved
 *  Fast Gauss Transform and Efficient Kernel Density Estimation},
 *  booktitle = {ICCV '03: Proceedings of the Ninth IEEE International
 *  Conference on Computer Vision}, year = {2003}, isbn =
 *  {0-7695-1950-4}, pages = {464}, publisher = {IEEE Computer
 *  Society}, address = {Washington, DC, USA}, }
 *
 *  LICENSE for FIGTREE V1.0 Copyright (c) 2002-2004, University of
 *  Maryland, College Park. All rights reserved.  The University of
 *  Maryland grants Licensee permission to use, copy, modify, and
 *  distribute FIGTREE ("Software) for any non-commercial, academic
 *  purpose without fee, subject to the following conditions: 1. Any
 *  copy or modification of this Software must include the above
 *  copyright notice and this license.  2. THE SOFTWARE AND ANY
 *  DOCUMENTATION ARE PROVIDED "AS IS"; THE UNIVERSITY OF MARYLAND
 *  MAKES NO WARRANTY OR REPRESENTATION THAT THE SOFTWARE WILL BE
 *  ERROR-FREE.  THE UNIVERSITY OF MARYLAND DISCLAIMS ANY AND ALL
 *  WARRANTIES, WHETHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED
 *  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 *  PARTICULAR PURPOSE, or NONINFRINGEMENT.  THIS DISCLAIMER OF
 *  WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS LICENSE.  IN NO
 *  EVENT WILL THE STATE OF MARYLAND, THE UNIVERSITY OF MARYLAND OR
 *  ANY OF THEIR RESPECTIVE OFFCERS, AGENTS, EMPLOYEES OR STUDENTS BE
 *  LIABLE FOR DIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL DAMAGES OR
 *  ANY KIND, INCLUDING, BUT NOT LIMITED TO, LOSS OF BUSINESS, WORK
 *  STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ANY AND ALL OTHER
 *  COMMERCIAL DAMAGES OR LOSSES, EVEN IF THE UNIVERSITY OF MARYLAND
 *  HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  3. The
 *  University of Maryland has no obligation to support, upgrade, or
 *  maintain the Software.  4. Licensee shall not use the name
 *  University of Maryland or any adaptation of the University of
 *  Maryland or any of its service or trade marks for any commercial
 *  purpose, including, but not limited to, publicity and advertising,
 *  without prior written consent of the University of Maryland.
 *  5. Licensee is responsible for complying with United States export
 *  control laws, including the Export Administration Regulations and
 *  International Traffic in Arms Regulations, and with United States
 *  boycott laws.  In the event a license is required under export
 *  control laws and regulations as a condition of exporting this
 *  Software outside the United States or to a foreign national in the
 *  United State, Licensee will be responsible for obtaining said
 *  license.  6. This License will terminate automatically upon
 *  Licensee's failure to correct its breach of any term of this
 *  License within thirty (30) days of receiving notice of its breach.
 *  7. Report bugs to: Changjiang Yang (yangcj@umiacs.umd.edu) and
 *  Ramani Duraiswami (ramani@umiacs.umd.edu), Department of Computer
 *  Science, University of Maryland, College Park.
 *
 *  Submit a request for a commercial license to: Jim Poulos, III
 *  Office of Technology Commercialization University of Maryland 6200
 *  Baltimore Avenue, Suite 300 Riverdale, Maryland 20737-1054
 *  301-403-2711 tel 301-403-2717 fax
 *
 *  @author Changjiang Yang (modified by Dongryeol Lee for FastLib usage)
 */

#ifndef ORIGINAL_IFGT_H
#define ORIGINAL_IFGT_H

#include "fastlib/fastlib.h"

class OriginalIFGT {

 private:

  ////////// Private Member Variables //////////

  /** @brief The module pointing to the parameters for execution. */
  struct datanode *module_;

  /** @brief Dimensionality of the points. */
  int dim_;

  /** @brief The number of reference points. */
  int num_reference_points_;

  /** @brief The column-oriented query dataset. */
  Matrix query_set_;

  /** @brief The column-oriented reference dataset. */
  Matrix reference_set_;

  /** @brief The weights associated with each reference point. */
  Vector reference_weights_;

  /** @brief The bandwidth */
  double bandwidth_;

  /** @brief Squared bandwidth */
  double bandwidth_sq_;

  /** @brief Bandwidth multiplied by \f$\sqrt{2}\f$ for modifying the
   *         original code to do computation for the Gaussian kernel
   *         \f$e^{-\frac{x^2}{2h^2}} \f$, rather than the alternative
   *         Gaussian kernel \f$e^{-\frac{x^2}{h^2}}\f$ used by the
   *         original code.
   */
  double bandwidth_factor_;

  /** @brief The desired absolute error precision. */
  double epsilon_;

  /** @brief The truncation order. */
  int pterms_;

  /** @brief The total number of coefficients */
  int total_num_coeffs_;
  
  /** @brief The coefficients weighted by reference_weights_ */
  Matrix weighted_coeffs_;

  /** @brief The unweighted coefficients */
  Matrix unweighted_coeffs_;

  /** @brief The number of clusters desired for preprocessing */
  int num_cluster_desired_;

  /** @brief If the distance between a query point and the cluster
   *         centroid is more than this quanity times the bandwidth,
   *         then the kernel sum contribution of the cluster is
   *         presumed to be zero.
   */
  double cut_off_radius_;

  /** @brief The maximum radius of the cluster among those generated.
   */
  double max_radius_cluster_;

  /** @brief The set of cluster centers
   */
  Matrix cluster_centers_;

  /** @brief The reference point index that is being used for the
   *         center of the clusters during K-center algorithm.
   */
  ArrayList<int> index_during_clustering_;

  /** @brief The i-th position of this vector tells the cluster number
   *         to which the i-th reference point belongs.
   */
  ArrayList<int> cluster_index_;

  /** @brief The i-th position of this vector tells the radius of the i-th
   *         cluster.
   */
  Vector cluster_radii_;

  /** @brief The number of reference points owned by each cluster.
   */
  ArrayList<int> num_reference_points_in_cluster_;

  /** @brief This will hold the final computed densities.
   */
  Vector densities_;

  ////////// Private Member Functions //////////
  void TaylorExpansion() {
      
    Vector tmp_coeffs;
    tmp_coeffs.Init(total_num_coeffs_);
    
    ComputeUnweightedCoeffs_(tmp_coeffs);
    
    ComputeWeightedCoeffs_(tmp_coeffs);
    
    return;	
  }

  void ComputeUnweightedCoeffs_(Vector &taylor_coeffs) {
	
    ArrayList<int> heads;
    heads.Init(dim_ + 1);
    ArrayList<int> cinds;
    cinds.Init(total_num_coeffs_);
    
    for (int i = 0; i < dim_; i++) {
      heads[i] = 0;
    }
    heads[dim_] = INT_MAX;
    
    cinds[0] = 0;
    taylor_coeffs[0] = 1.0;
    
    for(int k = 1, t = 1, tail = 1; k < pterms_; k++, tail = t) {
      for(int i = 0; i < dim_; i++) {
	int head = heads[i];
	heads[i] = t;
	for(int j = head; j < tail; j++, t++) {
	  cinds[t] = (j < heads[i+1]) ? cinds[j] + 1 : 1;
	  taylor_coeffs[t] = 2.0 * taylor_coeffs[j];
	  taylor_coeffs[t] /= (double) cinds[t];
	}
      }
    }
    return;    
  }

  void ComputeWeightedCoeffs_(Vector &taylor_coeffs) {
    
    Vector dx;
    dx.Init(dim_);
    Vector prods;
    prods.Init(total_num_coeffs_);
    ArrayList<int> heads;
    heads.Init(dim_);
    
    // initialize coefficients for all clusters to be zero.
    weighted_coeffs_.SetZero();
    unweighted_coeffs_.SetZero();
    
    for(int n = 0; n < num_reference_points_; n++) {
      
      int ix2c = cluster_index_[n];
      double sum = 0.0;
      
      for(int i = 0; i < dim_; i++) {
	dx[i] = (reference_set_.get(i, n) - cluster_centers_.get(i, ix2c)) / 
	  bandwidth_factor_;

	sum -= dx[i] * dx[i];
	heads[i] = 0;
      }
      
      prods[0] = exp(sum);
      for(int k = 1, t = 1, tail = 1; k < pterms_; k++, tail = t) {
	
	for (int i = 0; i < dim_; i++) {
	  int head = heads[i];
	  heads[i] = t;
	  for(int j = head; j < tail; j++, t++)
	    prods[t] = dx[i] * prods[j];
	} // for i
      } // for k
      
      // compute the weighted coefficients and unweighted coefficients.
      for(int i = 0; i < total_num_coeffs_; i++) {
	weighted_coeffs_.set(i, ix2c, weighted_coeffs_.get(i, ix2c) +
			     reference_weights_[n] * prods[i]);
	unweighted_coeffs_.set(i, ix2c, unweighted_coeffs_.get(i, ix2c) +
			       prods[i]);
      }
      
    }// for n
    
    // normalize by the Taylor coefficients.
    for(int k = 0; k < num_cluster_desired_; k++) {
      for(int i = 0; i < total_num_coeffs_; i++) {
	weighted_coeffs_.set(i, k, weighted_coeffs_.get(i, k) * 
			     taylor_coeffs[i]);
	unweighted_coeffs_.set(i, k, unweighted_coeffs_.get(i, k) * 
			       taylor_coeffs[i]);
      }
    }
    return;
  }

  /** @brief Compute the center and the radius of each cluster.
   *
   *  @return The maximum radius of the cluster among generated clusters.
   */
  double ComputeCenters() {

    // set cluster max radius to zero
    max_radius_cluster_ = 0;

    // clear all centers.
    cluster_centers_.SetZero();
    
    // Compute the weighted centroid for each cluster.
    for(int j = 0; j < dim_; j++) {
      for(int i = 0; i < num_reference_points_; i++) {
	
	cluster_centers_.set(j, cluster_index_[i],
			     cluster_centers_.get(j, cluster_index_[i]) +
			     reference_set_.get(j, i));
      }
    }
    
    for(int j = 0; j < dim_; j++) {
      for(int i = 0; i < num_cluster_desired_; i++) {
	cluster_centers_.set(j, i, cluster_centers_.get(j, i) /
			     num_reference_points_in_cluster_[i]);
      }
    }
    
    // Now loop through and compute the radius of each cluster.
    cluster_radii_.SetZero();
    for(int i = 0; i < num_reference_points_; i++) {
      Vector reference_pt;
      reference_set_.MakeColumnVector(i, &reference_pt);

      // the index of the cluster this reference point belongs to.
      int cluster_id = cluster_index_[i];
      Vector center;
      cluster_centers_.MakeColumnVector(cluster_id, &center);
      cluster_radii_[cluster_id] = 
	std::max(cluster_radii_[cluster_id], 
		 sqrt(la::DistanceSqEuclidean(reference_pt, center)));
      max_radius_cluster_ =
	std::max(max_radius_cluster_, cluster_radii_[cluster_id]);
    }

    return max_radius_cluster_;
  }

  /** @brief Perform the farthest point clustering algorithm on the 
   *         reference set.
   */
  double KCenterClustering() {
    
    Vector distances_to_center;
    distances_to_center.Init(num_reference_points_);
    
    // randomly pick one node as the first center.
    srand( (unsigned)time( NULL ) );
    int ind = rand() % num_reference_points_;
    
    // add the ind-th node to the first center.
    index_during_clustering_[0] = ind;
    Vector first_center;
    reference_set_.MakeColumnVector(ind, &first_center);
    
    // compute the distances from each node to the first center and
    // initialize the index of the cluster ID to zero for all
    // reference points.
    for(int j = 0; j < num_reference_points_; j++) {
      Vector reference_point;
      reference_set_.MakeColumnVector(j, &reference_point);
      
      distances_to_center[j] = (j == ind) ? 
	0.0:la::DistanceSqEuclidean(reference_point, first_center);
      cluster_index_[j] = 0;
    }
    
    // repeat until the desired number of clusters is reached.
    for(int i = 1; i < num_cluster_desired_; i++) {
      
      // Find the reference point that is farthest away from the
      // current center.
      ind = IndexOfLargestElement(distances_to_center);
      
      // Add the ind-th node to the centroid list.
      index_during_clustering_[i] = ind;
      
      // Update the distances from each point to the current center.
      Vector center;
      reference_set_.MakeColumnVector(ind, &center);
      
      for (int j = 0; j < num_reference_points_; j++) {
	Vector reference_point;
	reference_set_.MakeColumnVector(j, &reference_point);
	double d = (j == ind)? 
	  0.0:la::DistanceSqEuclidean(reference_point, center);
	
	if (d < distances_to_center[j]) {
	  distances_to_center[j] = d;
	  cluster_index_[j] = i;
	}
      }
    }
    
    // Find the maximum radius of the k-center algorithm.
    ind = IndexOfLargestElement(distances_to_center);
    
    double radius = distances_to_center[ind];
    
    
    for(int i = 0; i < num_cluster_desired_; i++) {
      num_reference_points_in_cluster_[i] = 0;
    }
    // tally up the number of reference points for each cluster.
    for (int i = 0; i < num_reference_points_; i++) {
      num_reference_points_in_cluster_[cluster_index_[i]]++;
    }
    
    return sqrt(radius);  
  }

  /** @brief Return the index whose position in the vector contains
   *         the largest element.
   */
  int IndexOfLargestElement(const Vector &x) {
    
    int largest_index = 0;
    double largest_quantity = -DBL_MAX;
    
    for(int i = 0; i < x.length(); i++) {
      if(largest_quantity < x[i]) {
	largest_quantity = x[i];
	largest_index = i;
      }
    }
    return largest_index;
  }

  /** 
   * Normalize the density estimates after the unnormalized sums have
   * been computed 
   */
  void NormalizeDensities() {
    double norm_const = pow(2 * math::PI * bandwidth_sq_, 
			    query_set_.n_rows() / 2.0) *
      reference_set_.n_cols();

    for(index_t q = 0; q < query_set_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
  }

  void IFGTChooseTruncationNumber_() {	

    double rx = max_radius_cluster_;
    double max_diameter_of_the_datasets = sqrt(dim_);
    
    double two_h_square = bandwidth_factor_ * bandwidth_factor_;
    
    double r = min(max_diameter_of_the_datasets, 
		   bandwidth_factor_ * sqrt(log(1 / epsilon_)));
    
    int p_ul=300;
    
    double rx_square = rx * rx;
    
    double error = 1;
    double temp = 1;
    int p = 0;
    while((error > epsilon_) & (p <= p_ul)) {
      p++;
      double b = min(((rx + sqrt((rx_square) + (2 * p * two_h_square))) / 2),
		     rx + r);
      double c = rx - b;
      temp = temp * (((2 * rx * b) / two_h_square) / p);
      error = temp * (exp(-(c * c) / two_h_square));			
    }	
    
    // update the truncation order.
    pterms_ = p;

    // update the cut-off radius
    cut_off_radius_ = r;
    
  }
  
  void IFGTChooseParameters_(int max_num_clusters) {
    
    // for references and queries that fit in the unit hypercube, this
    // assumption is true, but for general case it is not.
    double max_diamater_of_the_datasets = sqrt(dim_);
    
    double two_h_square = bandwidth_factor_ * bandwidth_factor_;

    // The cut-off radius.
    double r = min(max_diamater_of_the_datasets, 
		   bandwidth_factor_ * sqrt(log(1 / epsilon_)));
    
    // Upper limit on the truncation number.
    int p_ul=200; 
    
    num_cluster_desired_ = 1;
    
    double complexity_min=1e16;
    double rx;

    for(int i = 0; i < max_num_clusters; i++){
     
      // Compute an estimate of the maximum cluster radius.
      rx = pow((double) i + 1, -1.0 / (double) dim_);
      double rx_square = rx * rx;

      // An estimate of the number of neighbors.
      double n = std::min(i + 1.0, pow(r / rx, (double) dim_));
      double error = 1;
      double temp = 1;
      int p = 0;

      // Choose the truncation order.
      while((error > epsilon_) & (p <= p_ul)) {
	p++;
	double b = 
	  std::min(((rx + sqrt((rx_square) + (2 * p * two_h_square))) / 2.0),
		   rx + r);
	double c = rx - b;
	temp = temp * (((2 * rx * b) / two_h_square) / p);
	error = temp * (exp(-(c * c) / two_h_square));
      }
      double complexity = (i + 1) + log((double) i + 1) + 
	((1 + n) * math::BinomialCoefficient(p - 1 + dim_, dim_));
	
      if(complexity < complexity_min) {
	complexity_min = complexity;
	num_cluster_desired_ = i + 1;
	pterms_ = p;
      }
    }    
  }

 public:

  ////////// Constructor/Destructor //////////

  /** @brief Constructor
   */
  OriginalIFGT() {}

  /** @brief Destructor
   */
  ~OriginalIFGT() {}

  
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

  ////////// User-leverl Functions //////////

  void Init(Matrix &queries, Matrix &references, struct datanode *module_in) {
    
    // set module to the incoming one.
    module_ = module_in;
    
    // set dimensionality
    dim_ = references.n_rows();
        
    // set up query set and reference set.
    query_set_.Copy(queries);
    reference_set_.Copy(references);
    num_reference_points_ = reference_set_.n_cols();
    
    // By default, the we do uniform weights only
    reference_weights_.Init(reference_set_.n_cols());
    reference_weights_.SetAll(1);

    // initialize density estimate vector
    densities_.Init(query_set_.n_cols());
    
    // A "hack" such that the code uses the proper Gaussian kernel.
    bandwidth_ = fx_param_double_req(module_, "bandwidth");
    bandwidth_sq_ = bandwidth_ * bandwidth_;
    bandwidth_factor_ = sqrt(2) * bandwidth_;
    
    // Read in the desired absolute error accuracy.
    epsilon_ = fx_param_double(module_, "absolute_error", 0.1);

    // This is the upper limit on the number of clusters.
    int cluster_limit = (int) ceilf(20.0 * sqrt(dim_) / sqrt(bandwidth_));
    
    VERBOSE_MSG("Automatic parameter selection phase...\n");

    fx_timer_start(module_, "ifgt_kde_preprocess");
    IFGTChooseParameters_(cluster_limit);
    VERBOSE_MSG("Chose %d clusters...\n", num_cluster_desired_);
    VERBOSE_MSG("Tentatively chose %d truncation order...\n", pterms_);

    // Allocate spaces for storing coefficients and clustering information.
    cluster_centers_.Init(dim_, num_cluster_desired_);
    index_during_clustering_.Init(num_cluster_desired_);
    cluster_index_.Init(num_reference_points_);
    cluster_radii_.Init(num_cluster_desired_);
    num_reference_points_in_cluster_.Init(num_cluster_desired_);    
    
    VERBOSE_MSG("Now clustering...\n");

    // Divide the source space into num_cluster_desired_ parts using
    // K-center algorithm
    max_radius_cluster_ = KCenterClustering();
    
    // computer the center of the sources
    ComputeCenters();

    // Readjust the truncation order based on the actual clustering result.
    IFGTChooseTruncationNumber_();
    // pd = C_dim^(dim+pterms-1)
    total_num_coeffs_ = 
      (int) math::BinomialCoefficient(pterms_ + dim_ - 1, dim_);
    weighted_coeffs_.Init(total_num_coeffs_, num_cluster_desired_);
    unweighted_coeffs_.Init(total_num_coeffs_, num_cluster_desired_);

    printf("Maximum radius generated in the cluster: %g...\n",
	   max_radius_cluster_);
    printf("Truncation order updated to %d after clustering...\n", 
	   pterms_);

    // Compute coefficients.    
    VERBOSE_MSG("Now computing Taylor coefficients...\n");
    TaylorExpansion();
    VERBOSE_MSG("Taylor coefficient computation finished...\n");
    fx_timer_stop(module_, "ifgt_kde_preprocess");
    printf("Preprocessing step finished...\n");
  }

  void Compute() {
    
    printf("Starting the original IFGT-based KDE computation...\n");

    fx_timer_start(module_, "original_ifgt_kde_compute");

    Vector dy;
    dy.Init(dim_);
    
    Vector tempy;
    tempy.Init(dim_);
    
    Vector prods;
    prods.Init(total_num_coeffs_);
    
    ArrayList<int> heads;
    heads.Init(dim_);
    
    // make sure the sum for each query point starts at zero.
    densities_.SetZero();
    
    for(int m = 0; m < query_set_.n_cols(); m++) {	
      
      // loop over each cluster and evaluate Taylor expansions.
      for(int kn = 0; kn < num_cluster_desired_; kn++) {
	
	double sum2 = 0.0;
	
	// compute the ratio of the squared distance between each query
	// point and each cluster center to the bandwidth factor.
	for (int i = 0; i < dim_; i++) {
	  dy[i] = (query_set_.get(i, m) - cluster_centers_.get(i, kn)) / 
	    bandwidth_factor_;
	  sum2 += dy[i] * dy[i];
	}
	
	// If the ratio is greater than the cut-off, this cluster's
	// contribution is ignored.
	if (sum2 > (cut_off_radius_ + cluster_radii_[kn]) /
	    (bandwidth_factor_ * bandwidth_factor_)) {
	  continue;
	}
	
	for(int i = 0; i < dim_; i++) {
	  heads[i] = 0;
	}
	
	prods[0] = exp(-sum2);		
	for(int k = 1, t = 1, tail = 1; k < pterms_; k++, tail = t) {
	  for (int i = 0; i < dim_; i++) {
	    int head = heads[i];
	    heads[i] = t;
	    for(int j = head; j < tail; j++, t++)
	      prods[t] = dy[i] * prods[j];
	  } // for i
	}// for k
	
	for(int i = 0; i < total_num_coeffs_; i++) {
	  densities_[m] += weighted_coeffs_.get(i, kn) * prods[i];
	}
	
      } // for each cluster
    } //for each query point

    // normalize density estimates
    NormalizeDensities();

    fx_timer_stop(module_, "original_ifgt_kde_compute");
    printf("Computation finished...\n");
    return;
  }

  void PrintDebug() {
    
    FILE *stream = stdout;
    const char *fname = NULL;
    
    if((fname = fx_param_str(module_, "ifgt_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < query_set_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
