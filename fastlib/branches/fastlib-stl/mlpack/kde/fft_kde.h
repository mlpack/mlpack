/** @file fft_kde.h
 *
 *  This file contains an implementation of kernel density estimation
 *  using multidimensional fast Fourier transform for a linkable
 *  library component. This algorithm by design supports only the
 *  Gaussian kernel with the fixed-bandwidth. The optimal bandwidth
 *  cross-validation routine is not provided in this library.
 *
 *  For more details on mathematical derivations, please take a look at
 *  the following paper:
 *
 *  Article{wand94,
 *   Author = "M. P. Wand",
 *   Title = "{Fast Computation of Multivariate Kernel Estimators}",
 *   Journal = "Journal of Computational and Graphical Statistics",
 *   Year = "1994"
 *  }
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */
#ifndef FFT_KDE_H
#define FFT_KDE_H

#include <fastlib/fastlib.h>
#include <armadillo>
  
/** constant TAU */
#define TAU 4.0

/** @brief A computation class for FFT based kernel density estimation
 *
 *  This class is only inteded to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    FFTKde fft_kde;
 *    struct datanode* fft_kde_module;
 *    Vector results;
 *
 *    fft_kde_module = fx_submodule(NULL, "kde", "fft_kde_module");
 *    fft_kde.Init(queries, references, fft_kde_module);
 *    fft_kde.Compute();
 *
 *    // important to make sure that you don't call Init on results!
 *    fft_kde.get_density_estimates(&results);
 *  @endcode
 */
class FFTKde {

 private:

  ////////// Private Class Definitions //////////

  /** @brief Complex number - composed of real and imaginary parts */
  struct complex {

    /** @brief Real part */
    double real;

    /** @brief Imaginary part */
    double imag;
  };

  ////////// Private Member Variables //////////

  /** pointer to the module holding the relevant parameters */
  struct datanode *module_;



  /** query dataset */
  arma::mat qset_;

  /** reference dataset */
  arma::mat rset_;

  /** kernel */
  GaussianKernel kernel_;

  /** computed densities */
  arma::vec densities_;

  /** number of grid points along each dimension */
  int m_;

  /** number of points along each dimension in the zero padded */
  std::vector<int> size_;
  
  /** minimum coordinate along each dimension */
  arma::vec mincoords_;

  /** minimum indices along each dimension */
  std::vector<int> minindices_;

  /** maximum coordinate along each dimension */
  arma::vec maxcoords_;
  
  /** difference between min and max along each dimension */
  arma::vec diffcoords_;

  /** size of grid along each dimension */
  arma::vec gridsizes_;

  /** kernel weights along each dimension */
  std::vector<int>  kernelweights_dims_;

  /** total number of grid points */
  int numgridpts_;
  
  /** grid box volume */
  double gridbinvolume_;

  /** discretized dataset storing the assigned kernel weights */
  arma::vec discretized_;

  int nyquistnum_;

  arma::vec d_fnyquist_;
  
  arma::vec k_fnyquist_;
  
  arma::vec kernelweights_;
  
  /**
   * Do a Fourier transform of an array of N complex numbers separated by
   * steps of (complex) size skip.  The array f should be of length 2N*skip
   * and N must be a power of 2.  Forward determines whether to do a
   * forward transform (1) or an inverse one (-1)
   */
  void fftc1(double *f, int N, int skip, int forward) {

    int b, index1, index2, trans_size, trans;
    double pi2 = 4. * asin(1.);
    
    // used in recursive formula for Re(W^b) and Im(W^b)
    double pi2n, cospi2n, sinpi2n;
    
    // wk = W^k = e^(2 pi i b/N) in the Danielson-Lanczos formula for a
    // transform of length N
    struct complex wb;
    
    // buffers for implementing recursive formulas
    struct complex temp1, temp2;

    // treat f as an array of N complex numbers
    struct complex *c = (struct complex *)f;

    // Place the elements of the array c in bit-reversed order
    for(index1 = 1, index2 = 0; index1 < N; index1++) {
      
      // to find the next bit reversed array index subtract leading 1's from
      // index2
      for(b = N / 2; index2 >= b; b /= 2) {
	index2 -= b;
      }
      
      // Next replace the first 0 in index2 with a 1 and this gives the 
      // correct next value
      index2 += b;
      
      // swap each pair only the first time it is found
      if(index2 > index1) {
	temp1 = c[index2 * skip];
	c[index2 * skip] = c[index1 * skip];
	c[index1 * skip] = temp1;
      }
    }

    // Next perform successive transforms of length 2,4,...,N using the
    // Danielson-Lanczos formula

    // trans_size = size of transform being computed
    for(trans_size = 2; trans_size <= N; trans_size *= 2) {
      
      // +- 2 pi/trans_size
      pi2n = forward * pi2 / (double)trans_size;
      
      // Used to calculate W^k in D-L formula
      cospi2n = cos(pi2n);
      sinpi2n = sin(pi2n);
      
      // Initialize W^b for b=0
      wb.real = 1.;
      wb.imag = 0.;
      
      // Step over half of the elements in the transform
      for(b = 0; b < trans_size / 2; b++) {
	
	// Iterate over all transforms of size trans_size to be computed
	for(trans = 0; trans < N / trans_size; trans++) {

	  // Index of element in first half of transform being computed
	  index1 = (trans * trans_size + b) * skip;

	  // Index of element in second half of transform being computed
	  index2 = index1 + trans_size / 2 * skip;
	  temp1 = c[index1];
	  temp2 = c[index2];

	  // implement D-L formula
	  c[index1].real = temp1.real + wb.real * temp2.real - 
	    wb.imag * temp2.imag;
	  c[index1].imag = temp1.imag + wb.real * temp2.imag + 
	    wb.imag * temp2.real;
	  c[index2].real = temp1.real - wb.real * temp2.real + 
	    wb.imag * temp2.imag;
	  c[index2].imag = temp1.imag - wb.real * temp2.imag - 
	    wb.imag * temp2.real;
	}
	temp1 = wb;

	// Real part of e^(2 pi i b/trans_size) used in D-L formula
	wb.real = cospi2n * temp1.real - sinpi2n * temp1.imag;
	
	// Imaginary part of e^(2 pi i b/trans_size) used in D-L formula
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real;
      }
    }
    
    // For an inverse transform divide by the number of grid points
    if(forward<0) {
      for(index1 = 0; index1 < skip * N; index1 += skip) {
	c[index1].real /= N;
	c[index1].imag /= N;
      }
    }
  }

  /**
   * Do a Fourier transform of an ndims dimensional array of complex numbers
   * Array dimensions are given by size[0],...,size[ndims-1]. Note that these 
   * are sizes of complex arrays. The array f should be of length 
   * 2*size[0]*...*size[ndims-1] and all sizes must be powers of 2.
   * Forward determines whether to do a forward transform (1) or an inverse 
   * one(-1)
   */
  void fftcn(double *f, int ndims, std::vector<int>& size, int forward) {

    // These determine where to begin successive transforms and the skip 
    // between their elements (see below)
    int planesize = 1, skip = 1;

    // Total size of the ndims dimensional array
    int totalsize = 1;
    
    // determine total size of array
    for(index_t dim = 0; dim < ndims; dim++) {
      totalsize *= size[dim];
    }

    // loop over dimensions
    for(index_t dim = ndims - 1; dim >= 0; dim--) {
      
      // planesize = Product of all sizes up to and including size[dim] 
      planesize *= size[dim];

      // Take big steps to begin loops of transforms 
      for(index_t i = 0; i < totalsize; i += planesize) {
	
	// Skip sets the number of transforms in between big steps as well as 
	// the skip between elements
	for(index_t j = 0; j < skip; j++) {
	  
	  // 1-D Fourier transform. (Factor of two converts complex index to 
	  // double index.)
	  fftc1(f + 2 * (i + j), size[dim], skip, forward);
	}
      }
      // Skip = Product of all sizes up to (but not including) size[dim]
      skip *= size[dim];
    }
  }

  /**
   * Do a Fourier transform of an array of N real numbers
   * N must be a power of 2
   * Forward determines whether to do a forward transform (>=0) or an inverse 
   * one(<0)
   */
  void fftr1(double *f, int N, int forward) {

    int b;
    
    // pi2n = 2 Pi/N
    double pi2n = 4. * asin(1.) / N, cospi2n = cos(pi2n), sinpi2n = sin(pi2n);
    
    // wb = W^b = e^(2 pi i b/N) in the Danielson-Lanczos formula for a 
    // transform of length N
    struct complex wb;
    
    // Buffers for implementing recursive formulas
    struct complex temp1, temp2;
    
    // Treat f as an array of N/2 complex numbers
    struct complex *c = (struct complex *)f;
    
    // Do a transform of f as if it were N/2 complex points
    if(forward == 1) {
      fftc1(f, N / 2, 1, 1);
    }

    // initialize W^b for b = 0
    wb.real = 1.;
    wb.imag = 0.;

    // Loop over elements of transform. See documentation for these formulas 
    for(b = 1; b < N / 4; b++) {

      temp1 = wb;

      // Real part of e^(2 pi i b/N) used in D-L formula 
      wb.real = cospi2n * temp1.real - sinpi2n * temp1.imag;
      
      // Imaginary part of e^(2 pi i b/N) used in D-L formula
      wb.imag = cospi2n * temp1.imag + sinpi2n * temp1.real;
      temp1 = c[b];
      temp2 = c[N / 2 - b];
      c[b].real = .5 * (temp1.real + temp2.real + forward * wb.real * 
			(temp1.imag + temp2.imag) + wb.imag * 
			(temp1.real - temp2.real));
      c[b].imag = .5 * (temp1.imag-temp2.imag - forward * wb.real * 
			(temp1.real - temp2.real) + wb.imag * 
			(temp1.imag + temp2.imag));
      c[N/2-b].real = .5 * (temp1.real + temp2.real - forward * wb.real * 
			    (temp1.imag + temp2.imag) - wb.imag * 
			    (temp1.real - temp2.real));
      c[N/2-b].imag = .5 * (-temp1.imag + temp2.imag - forward * wb.real * 
			    (temp1.real - temp2.real) + wb.imag * 
			    (temp1.imag + temp2.imag));
    }

    temp1 = c[0];

    // set b = 0 term in transform
    c[0].real = temp1.real+temp1.imag;
    
    // put b = N / 2 term in imaginary part of first term
    c[0].imag = temp1.real-temp1.imag;
    
    if(forward == -1) {
      c[0].real *= .5;
      c[0].imag *= .5;
      fftc1(f, N / 2, 1, -1);
    }
  }

  /**
   * Do a Fourier transform of an ndims dimensional array of real numbers
   * Array dimensions are given by size[0],...,size[ndims-1]. All sizes must 
   * be powers of 2. The (complex) nyquist frequency components are stored in 
   * fnyquist[size[0]][size[1]]...[2*size[ndims-2]]
   * Forward determines whether to do a forward transform (1) or an inverse 
   * one (-1)
   */
  void fftrn(double *f, double *fnyquist, int ndims, std::vector<int>& size, int forward) {

    int i, j, b;

    // Positions in the 1-d arrays of points labeled by indices 
    // (i0,i1,...,i(ndims-1)); indexneg gives the position in the array of 
    // the corresponding negative frequency
    int index,indexneg = 0;
    int stepsize; // Used in calculating indexneg

    // The size of the last dimension is used often enough to merit its own 
    // name.
    int N = size[ndims - 1];
    
    // pi2n = 2Pi / N
    double pi2n = 4. * asin(1.) / N, cospi2n = cos(pi2n), sinpi2n = sin(pi2n);

    // wb = W^b = e^(2 pi i b/N) in the Danielson-Lanczos formula for a 
    // transform of length N
    struct complex wb; 

    // Buffers for implementing recursive formulas
    struct complex temp1, temp2;

    // Treat f and fnyquist as arrays of complex numbers
    struct complex *c = (struct complex *)f, 
      *cnyquist = (struct complex *)fnyquist;

    // Total number of complex points in array
    int totalsize = 1;

    // Indices for looping through array
    std::vector<int> indices;
    indices.reserve(ndims);
    
    // Set size[] to be the sizes of f viewed as a complex array
    size[ndims - 1] /= 2;

    for(i = 0; i < ndims; i++) {
      totalsize *= size[i];
      indices[i] = 0;
    }
    
    // forward transform
    if(forward == 1) {

      // Do a transform of f as if it were N/2 complex points 
      fftcn(f, ndims, size, 1);

      // Copy b=0 data into cnyquist so the recursion formulas below for b=0 
      // and cnyquist don't overwrite data they later need
      for(i = 0; i < totalsize / size[ndims - 1]; i++) {
	
	// Only copy points where last array index for c is 0
	cnyquist[i] = c[i * size[ndims - 1]];
      }
    }
    
    // Loop over all but last array index
    for(index = 0; index < totalsize; index += size[ndims-1]) {

      wb.real = 1.; /* Initialize W^b for b=0 */
      wb.imag = 0.;

      // Loop over elements of transform. See documentation for these formulas
      for(b = 1; b < N / 4; b++) {

	temp1 = wb;

	// Real part of e^(2 pi i b/N_real) used in D-L formula
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag;

	// Imaginary part of e^(2 pi i b/N_real) used in D-L formula
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real;

	temp1 = c[index + b];

	// Note that N-b is NOT the negative frequency for b. Only 
	// nonnegative b momenta are stored.
	temp2 = c[indexneg + N / 2 - b];

	c[index + b].real = .5 * (temp1.real + temp2.real + forward * wb.real * 
				  (temp1.imag + temp2.imag) + wb.imag * 
				  (temp1.real - temp2.real));
	c[index + b].imag = .5 * (temp1.imag - temp2.imag - forward * wb.real * 
				  (temp1.real - temp2.real) + wb.imag * 
				  (temp1.imag + temp2.imag));
	c[indexneg + N / 2 - b].real = .5 * (temp1.real + temp2.real - 
					     forward * 
					     wb.real * 
					     (temp1.imag + temp2.imag) - 
					     wb.imag * 
					     (temp1.real - temp2.real));
	c[indexneg + N / 2 - b].imag = .5 * (-temp1.imag + temp2.imag - 
					     forward * wb.real * 
					     (temp1.real - temp2.real) + 
					     wb.imag * 
					     (temp1.imag + temp2.imag));
      }
      temp1 = c[index];

      // Index is smaller for cnyquist because it doesn't have the last 
      // dimension
      temp2 = cnyquist[indexneg / size[ndims - 1]];

      // Set b=0 term in transform
      c[index].real = .5 * (temp1.real + temp2.real + forward * 
			    (temp1.imag + temp2.imag));
      c[index].imag = .5 * (temp1.imag - temp2.imag - forward * 
			    (temp1.real - temp2.real));

      // Set b=N/2 transform.
      cnyquist[indexneg / size[ndims - 1]].real = 
	.5 * (temp1.real + temp2.real - forward * (temp1.imag + temp2.imag));
      cnyquist[indexneg / size[ndims - 1]].imag = 
	.5 * (-temp1.imag + temp2.imag - forward * (temp1.real - temp2.real));
      
      // Find indices for positive and single index for negative frequency. 
      // In each dimension indexneg[j]=0 if index[j]=0, 
      // indexneg[j]=size[j]-index[j] otherwise.

      // amount to increment indexneg by as each individual index is 
      // incremented
      stepsize = size[ndims - 1];

      // If the rightmost indices are maximal reset them to 0. Indexneg goes 
      // from 1 to 0 in these dimensions
      for(j = ndims - 2; j >= 0 && indices[j] == size[j] - 1; j--) {
	indices[j] = 0;
	indexneg -= stepsize;
	stepsize *= size[j];
      }
      
      // If index[j] goes from 0 to 1 indexneg[j] goes from 0 to size[j]-1
      if(j >= 0 && indices[j] == 0) {
	indexneg += stepsize * (size[j] - 1);
      }
      // Otherwise increasing index[j] decreases indexneg by one unit.
      else {
	indexneg -= stepsize;
      }

      // This avoids writing outside the array bounds on the last pass 
      // through the array loop
      if(j >= 0) {
	indices[j]++;
      }
    } // End of i loop (over total array)
    
    // inverse transform
    if(forward == -1) {
      fftcn(f, ndims, size, -1);
    }
    
    // Give the user back the array size[] in its original condition
    size[ndims - 1] *= 2;

  }
  
  void assign_weights(int reference_pt_num, int level, double volume, int pos,
		      int skip) {
    if(level == -1) {
      discretized_[pos] += volume;
    }
    else {
      
      // Recurse in the right direction
      double coord = rset_(level, reference_pt_num);
      double leftgridcoord = mincoords_[level] + minindices_[level] *
	gridsizes_[level];
      double rightgridcoord = leftgridcoord + gridsizes_[level];
      double leftvolume = volume * (rightgridcoord - coord);
      double rightvolume = volume * (coord - leftgridcoord);
      int nextskip = size_[level] * skip;
      int nextleftpos = pos + skip * minindices_[level];
      
      if(leftvolume > 0.0) {
	assign_weights(reference_pt_num, level - 1, leftvolume, nextleftpos, 
		       nextskip);
      }

      if(rightvolume > 0.0) {
	assign_weights(reference_pt_num, level - 1, rightvolume, 
		       nextleftpos + skip, nextskip);
      }
    }
  }

  void retrieve_weights(int query_pt_num, double volume, int level, int pos, 
			int skip, double divfactor) {

    if(level == -1) {
      densities_[query_pt_num] += discretized_[pos] * volume / divfactor;
    }
    else {
      
      // Recurse in the right direction
      double coord = qset_(level, query_pt_num);
      double leftgridcoord = mincoords_[level] + minindices_[level] *
	gridsizes_[level];
      double rightgridcoord = leftgridcoord + gridsizes_[level];
      double leftvolume = volume * (rightgridcoord - coord);
      double rightvolume = volume * (coord - leftgridcoord);
      int nextskip = size_[level] * skip;
      int nextleftpos = pos + skip * minindices_[level];
      
      if(leftvolume > 0.0) {
	retrieve_weights(query_pt_num, leftvolume, level - 1, nextleftpos, 
			 nextskip, divfactor);
      }
      if(rightvolume > 0.0) {
	retrieve_weights(query_pt_num, rightvolume, level - 1, 
			 nextleftpos + skip, nextskip, divfactor);
      }
    }
  }

  /**
   * Query the normalized density for each query point.
   */
  void RetrieveDensities() {

    double normc = 
      (kernel_.CalcNormConstant(rset_.n_rows) * rset_.n_cols);

    for(index_t r = 0; r < qset_.n_cols; r++) {
      densities_[r] = 0.0;
      
      for(index_t d = 0; d < qset_.n_rows; d++) {
	minindices_[d] = (int) floor((qset_(d, r) - mincoords_[d])/
				     gridsizes_[d]);
      }
      retrieve_weights(r, 1.0, qset_.n_rows - 1, 0, 1, 
		       gridbinvolume_ * normc);
    }
  }

  void discretize_dataset() {

    // Temporary used to count the number of elements in the enlarged 
    // matrices for the kernel weights and bin counts. Also calculate the 
    // volume of each grid bin.
    numgridpts_ = 1;
    gridbinvolume_ = 1.0;
        
    double min, max;
    
    // Find the min/max in each coordinate direction, and calculate the grid
    // size in each dimension.
    for(index_t d = 0; d < qset_.n_rows; d++) {
      int possiblesample;
      min = DBL_MAX;
      max = -DBL_MAX;
      
      for(index_t r = 0; r < rset_.n_cols; r++) {
	double coord = rset_(d, r);
	if(coord > max)
	  max = coord;
	if(coord < min)
	  min = coord;
      }
      
      // Following Silverman's advice here
      mincoords_[d] = min;
      maxcoords_[d] = max;
      diffcoords_[d] = maxcoords_[d] - mincoords_[d];
      gridsizes_[d] = diffcoords_[d] / ((double) m_ - 1);
      gridbinvolume_ *= gridsizes_[d];

      // Determine how many kernel weight calculation to do for this 
      // dimension.
      kernelweights_dims_[d] = m_ - 1;
      possiblesample = (int) floor(TAU * sqrt(kernel_.bandwidth_sq()) / 
				   gridsizes_[d]);
      
      if(kernelweights_dims_[d] > possiblesample) {
	if(possiblesample == 0) {
	  possiblesample = 1;
	}
	kernelweights_dims_[d] = possiblesample;
      }

      // Wand p440: Need to calculate the actual dimension of the matrix
      // after the necessary 0 padding of the kernel weight matrix and the
      // bin count matrix.
      size_[d] = (int) ceil(log(m_ + kernelweights_dims_[d]) / log(2));
      size_[d] = 1 << size_[d];

      numgridpts_ *= size_[d];
    }

    // Allocate the memory for discretized grid count matrix and initialize 
    // it.
    discretized_.zeros(numgridpts_);

    double inv_gvolume = 1.0 / gridbinvolume_;
    
    // Now loop over each data and calculate the weights at each grid point.
    for(index_t r = 0; r < rset_.n_cols; r++) {

      // First locate the bin the data point falls into and identify it by
      // the lower grid coordinates.
      for(index_t d = 0; d < rset_.n_rows; d++) {
	minindices_[d] = (int) floor((rset_(d, r) - mincoords_[d])/
				     gridsizes_[d]);
      }

      // Assign the weights around the neighboring grid points due to this
      // data point. This results in 2^num_dims number of recursion per data
      // point.
      assign_weights(r, qset_.n_rows - 1, inv_gvolume, 0, 1);
    }

  }
  
  void Gaussify(double acc, double precalc, int level, int pos, int skip) {

    if(level == -1) {
      kernelweights_[pos] = exp(precalc * acc);
    }
    else {
      int half = kernelweights_dims_[level];
      int g;
      for(g = 0; g <= half; g++) {
	double addThis = g * gridsizes_[level];
	double newacc = acc + addThis * addThis;
	int newskip = skip * size_[level];
	
	Gaussify(newacc, precalc, level - 1, pos + skip * g, newskip);
	
	// If this is not the 0th frequency, then do the mirror image thingie.
	if(g != 0) {
	  Gaussify(newacc, precalc, level - 1,
		   pos + skip * (size_[level] - g), newskip);
	}
      }
    }
  }
  
  
 public:

  ////////// Constructor/Destructor //////////

  /** @brief Constructor - does not do anything */
  FFTKde() {}
  
  /** @brief Destructor - does not do anything */
  ~FFTKde() {}
  
  ////////// Getters/Setters //////////

  /** @brief Get the density estimates.
   *
   *  @param results An uninitialized vector which will be initialized with
   *                 the computed density estimates.
   */
  void get_density_estimates(arma::vec& results) {
    results = densities_;
  }

  /** @brief Initialize the FFT KDE object with the query and the
   *         reference datasets with the parameter lists.
   *
   *  @param qset The column-oriented query dataset.
   *  @param rset The column-oriented reference dataset.
   *  @param module_in The module containing the parameters for execution.
   */
  void Init(arma::mat& qset, arma::mat& rset, struct datanode *module_in) {
    
    // initialize module to the incoming one
    module_ = module_in;

    printf("Initializing FFT KDE...\n");
    fx_timer_start(module_, "fft_kde_init");

    // initialize the kernel and read in the number of grid points
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));
    m_ = fx_param_int(module_, "num_grid_pts_per_dim", 128);

    // set aliases to the query and reference datasets and initialize
    // query density sets
    qset_ = qset;
    densities_.set_size(qset_.n_cols);
    rset_ = rset;

    // initialize member variables.
    size_.reserve(qset_.n_rows);
    minindices_.reserve(rset_.n_rows);
    mincoords_.set_size(qset_.n_rows);
    maxcoords_.set_size(qset_.n_rows);
    diffcoords_.set_size(qset_.n_rows);
    gridsizes_.set_size(qset_.n_rows);
    kernelweights_dims_.reserve(qset_.n_rows);

    // set up the discretized grid for the reference dataset
    discretize_dataset();

    nyquistnum_ = 2 * numgridpts_ / size_[rset_.n_rows - 1];

    d_fnyquist_.set_size(nyquistnum_);
    k_fnyquist_.set_size(nyquistnum_);
    kernelweights_.set_size(numgridpts_);

    fx_timer_stop(module_, "fft_kde_init");
    printf("FFT KDE initialization completed...\n");
  }

  /** @brief Compute density estimates using FFT after initialization
   */
  void Compute() {

    printf("Computing FFT KDE...\n");
    fx_timer_start(module_, "fft_kde");

    // FFT the discretized bin count matrix.
    d_fnyquist_.zeros();
    k_fnyquist_.zeros();
    kernelweights_.zeros();
    fftrn(discretized_.memptr(), d_fnyquist_.memptr(), rset_.n_rows, 
	  size_, 1);

    // Calculate the required kernel weights at each grid point. This matrix
    // will be convolved with fourier transformed data set.
    double precalc = -0.5 / kernel_.bandwidth_sq();
    Gaussify(0.0, precalc, rset_.n_rows - 1, 0, 1);

    // FFT the kernel weight matrix.
    fftrn(kernelweights_.memptr(), k_fnyquist_.memptr(),
	  rset_.n_rows, size_, 1);

    // We need to invoke the convolution theorem for FFT here. Take each
    // corresponding complex number in kernelweights and discretized and do
    // an element-wise multiplication. Later, pass it to inverse fft function,
    // and we have our answer!
    for(index_t d = 0; d < numgridpts_; d += 2) {
      double real1 = discretized_[d];
      double complex1 = discretized_[d + 1];
      double real2 = kernelweights_[d];
      double complex2 = kernelweights_[d + 1];
      discretized_[d] = real1 * real2 - complex1 * complex2;
      discretized_[d + 1] = real1 * complex2 + complex1 * real2;
    }

    for(index_t d = 0; d < nyquistnum_; d += 2) {
      double real1 = d_fnyquist_[d];
      double complex1 = d_fnyquist_[d + 1];
      double real2 = k_fnyquist_[d];
      double complex2 = k_fnyquist_[d + 1];
      d_fnyquist_[d] = real1 * real2 - complex1 * complex2;
      d_fnyquist_[d + 1] = real1 * complex2 + complex1 * real2;
    }

    // Inverse FFT the elementwise multiplied matrix.
    fftrn(discretized_.memptr(), d_fnyquist_.memptr(), 
	  rset_.n_rows, size_, -1);

    // Retrieve the densities of each data point.
    RetrieveDensities();

    fx_timer_stop(module_, "fft_kde");
    printf("FFT KDE completed...\n");
  }

  /** @brief Output KDE results to a stream 
   *
   *  If the user provided "--fft_kde_output=" argument, then the
   *  output will be directed to a file whose name is provided after
   *  the equality sign.  Otherwise, it will be provided to the
   *  screen.
   */
  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(module_, "fft_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols; q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
