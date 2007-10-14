#ifndef FFT_KDE_H
#define FFT_KDE_H

/** 
 * computing kernel estimate using Fast Fourier Transform: I have
 * used multidimensional fast fourier transform called ffteasy
 */
template<typename TKernel>
class FFTKde {
  
 private:

  /**
   * Do a Fourier transform of an array of N complex numbers separated by
   * steps of (complex) size skip.  The array f should be of length 2N*skip
   * and N must be a power of 2.  Forward determines whether to do a
   * forward transform (1) or an inverse one (-1)
   */
  void fftc1(double f[], int N, int skip, int forward) {

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
	  index2 = index1 + trans_size/2*skip;
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
    if(forward<0.) {
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
  void fftcn(double f[], int ndims, int size[], int forward) {

    int i, j, dim;

    // These determine where to begin successive transforms and the skip 
    // between their elements (see below)
    int planesize = 1, skip = 1;

    // Total size of the ndims dimensional array
    int totalsize = 1;
    
    // determine total size of array
    for(dim = 0; dim < ndims; dim++) {
      totalsize *= size[dim];
    }
    
    // loop over dimensions
    for(dim = ndims - 1; dim >= 0; dim--) {
      
      // planesize = Product of all sizes up to and including size[dim] 
      planesize *= size[dim];

      // Take big steps to begin loops of transforms 
      for(i = 0; i < totalsize; i += planesize) {
	
	// Skip sets the number of transforms in between big steps as well as 
	// the skip between elements
	for(j = 0; j < skip; j++) {
	  
	  // 1-D Fourier transform. (Factor of two converts complex index to 
	  // double index.)
	  fftc1(f + 2 * (i + j), size[dim], skip, forward);
	}
	
	// Skip = Product of all sizes up to (but not including) size[dim]
	skip *= size[dim];
      }
    }
  }

  /**
   * Do a Fourier transform of an array of N real numbers
   * N must be a power of 2
   * Forward determines whether to do a forward transform (>=0) or an inverse 
   * one(<0)
   */
  void fftr1(double f[], int N, int forward) {

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
      fftc1(f, N/2, 1, 1);
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
      temp2 = c[N/2-b];
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
      fftc1(f, N/2, 1, -1);
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
  void fftrn(double f[], double fnyquist[], int ndims, int size[], 
	     int forward) {

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
    int *indices= (int *) malloc(ndims*sizeof(int));

    // Make sure memory was correctly allocated
    if(!indices) {
      printf("Error allocating memory in fftrn routine. Exiting.\n");
      exit(1);
    }
    
    // Set size[] to be the sizes of f viewed as a complex array
    size[ndims-1] /= 2;

    for(i = 0; i < ndims; i++) {
      totalsize *= size[i];
      indices[i] = 0;
    }
    
    // forward transform
    if(forward==1) {

      // Do a transform of f as if it were N/2 complex points 
      fftcn(f,ndims,size,1);

      // Copy b=0 data into cnyquist so the recursion formulas below for b=0 
      // and cnyquist don't overwrite data they later need
      for(i = 0; i < totalsize / size[ndims-1]; i++) {
	
	// Only copy points where last array index for c is 0
	cnyquist[i] = c[i*size[ndims-1]];
      }
    }
    
    // Loop over all but last array index
    for(index=0;index<totalsize;index+=size[ndims-1]) {

      wb.real = 1.; /* Initialize W^b for b=0 */
      wb.imag = 0.;

      // Loop over elements of transform. See documentation for these formulas
      for(b = 1; b < N / 4; b++) {

	temp1 = wb;

	// Real part of e^(2 pi i b/N_real) used in D-L formula
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag;

	// Imaginary part of e^(2 pi i b/N_real) used in D-L formula
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real;

	temp1 = c[index+b];

	// Note that N-b is NOT the negative frequency for b. Only 
	// nonnegative b momenta are stored.
	temp2 = c[indexneg+N/2-b];

	c[index+b].real = .5 * (temp1.real + temp2.real + forward * wb.real * 
				(temp1.imag + temp2.imag) + wb.imag * 
				(temp1.real - temp2.real));
	c[index+b].imag = .5 * (temp1.imag - temp2.imag - forward * wb.real * 
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
      temp2 = cnyquist[indexneg/size[ndims-1]];

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
      for(j = ndims - 2; indices[j] == size[j] - 1 && j >= 0; j--) {
	indices[j] = 0;
	indexneg -= stepsize;
	stepsize *= size[j];
      }
      
      // If index[j] goes from 0 to 1 indexneg[j] goes from 0 to size[j]-1
      if(indices[j]==0) {
	indexneg += stepsize*(size[j]-1);
      }
      // Otherwise increasing index[j] decreases indexneg by one unit.
      else {
	indexneg -= stepsize;
      }

      // This avoids writing outside the array bounds on the last pass 
      // through the array loop
      if(j>=0) {
	indices[j]++;
      }
    } // End of i loop (over total array)
    
    // inverse transform
    if(forward==-1) {
      fftcn(f,ndims,size,-1);
    }
    
    // Give the user back the array size[] in its original condition
    size[ndims-1] *= 2;

    // free up memory allocated for indices array
    free(indices);
  }
  
  void assignWeights(double *datapt, double *gridsizes,
		     double *mincoords, int *indices, int *enlarged_dims,
		     double *discretized, int level, double volume, int pos,
		     int skip) {
    if(level == -1) {
      discretized[pos] += volume;
    }
    else {
      
      // Recurse in the right direction
      double coord = datapt[level];
      double leftgridcoord = mincoords[level] + indices[level] *
	gridsizes[level];
      double rightgridcoord = leftgridcoord + gridsizes[level];
      double leftvolume = volume * (rightgridcoord - coord);
      double rightvolume = volume * (coord - leftgridcoord);
      int nextskip = enlarged_dims[level] * skip;
      int nextleftpos = pos + skip * indices[level];
      
      if(leftvolume > 0.0) {
	assignWeights(datapt, gridsizes, mincoords, indices,
		      enlarged_dims, discretized, level - 1, leftvolume,
		      nextleftpos, nextskip);
      }

      if(rightvolume > 0.0) {
	assignWeights(datapt, gridsizes, mincoords, indices,
		      enlarged_dims, discretized, level - 1, rightvolume,
		      nextleftpos + skip, nextskip);
      }
    }
  }

  void retrieveWeights(int dataptnum, double *datapt, int num_dims,
		       double *gridsizes, double *discretized, int *size,
		       int *indices, double *mincoords, double volume,
		       double *densities, int level, int pos, int skip,
		       double divfactor) {

    if(level == -1) {
      densities[dataptnum] += discretized[pos] * volume / divfactor;
    }
    else {
      
      // Recurse in the right direction
      double coord = datapt[level];
      double leftgridcoord = mincoords[level] + indices[level] *
	gridsizes[level];
      double rightgridcoord = leftgridcoord + gridsizes[level];
      double leftvolume = volume * (rightgridcoord - coord);
      double rightvolume = volume * (coord - leftgridcoord);
      int nextskip = size[level] * skip;
      int nextleftpos = pos + skip * indices[level];
      
      if(leftvolume > 0.0) {
	retrieveWeights(dataptnum, datapt, num_dims, gridsizes, discretized,
			size, indices, mincoords, leftvolume, densities,
			level - 1, nextleftpos, nextskip, divfactor);
      }
      if(rightvolume > 0.0) {
	retrieveWeights(dataptnum, datapt, num_dims, gridsizes, discretized,
			size, indices, mincoords, rightvolume, densities,
			level - 1, nextleftpos + skip, nextskip, divfactor);
      }
    }
  }

  double *retrieveDensities(double *dataset, int num_rows, int dim,
			    double *gridsizes, double *discretized,
			    int *size, double *mincoords, 
			    double gridbinvolume, double bandwidsqd) {

    double *densities = (double *) malloc(num_rows * sizeof(double));
    double normc = pow((2.0 * PI * bandwidsqd),((double)dim) / 2.0) * 
      num_rows;

    int r, d;
    int *minindices = (int *) malloc(dim * sizeof(int));
    
    for(r = 0; r < num_rows; r++) {
      densities[r] = 0.0;
      
      for(d = 0; d < dim; d++) {
	minindices[d] = floor((dataset[r * dim + d] - mincoords[d])/
			      gridsizes[d]);
      }
      retrieveWeights(r, dataset + r * dim, dim, gridsizes, discretized,
		      size, minindices, mincoords, 1.0, densities, dim - 1,
		      0.0, 1, gridbinvolume * normc);
    }
    free(minindices);
    return densities;
  }

  double *discretize_dataset(double *dataset, int num_rows, int dim,
			     double bandwidth, double *gridsizes,
			     double *mincoords, double *maxcoords,
			     double *diffcoords, int *kernelweights_dims,
			     int *enlarged_dims, int *numenlargedgridpts,
			     double *gridbinvolume) {

    // Temporary used to count the number of elements in the enlarged 
    // matrices for the kernel weights and bin counts. Also calculate the 
    // volume of each grid bin.
    int numengridpts = 1;
    double gvolume = 1.0;
    
    // This points to the discretized grids with each grid point storing the
    // counts.
    double *discretized;
    
    // Temporary index array to locate the bin for each data point.
    int *minindices = (int *) malloc(dim * sizeof(int));
    int r, d;
    double min, max;
    
    // Find the min/max in each coordinate direction, and calculate the grid
    // size in each dimension.
    for(d = 0; d < dim; d++) {
      int possiblesample;
      min = MAXDOUBLE;
      max = MINDOUBLE;
      
      for(r = 0; r < num_rows; r++) {
	double coord = dataset[r * dim + d];
	if(coord > max)
	  max = coord;
	if(coord < min)
	  min = coord;
      }
      
      // Following Silverman's advice here
      mincoords[d] = min;
      maxcoords[d] = max;
      diffcoords[d] = maxcoords[d] - mincoords[d];
      gridsizes[d] = diffcoords[d] / ((double) M - 1);
      gvolume *= gridsizes[d];

      // Determine how many kernel weight calculation to do for this 
      // dimension.
      kernelweights_dims[d] = M - 1;
      possiblesample = floor(TAU * bandwidth / gridsizes[d]);
      
      if(kernelweights_dims[d] > possiblesample) {
	if(possiblesample == 0)
	  possiblesample = 1;
	kernelweights_dims[d] = possiblesample;
      }

      // Wand p440: Need to calculate the actual dimension of the matrix
      // after the necessary 0 padding of the kernel weight matrix and the
      // bin count matrix.
      enlarged_dims[d] = ceil(log(M + kernelweights_dims[d]) / log(2));
      enlarged_dims[d] = 1 << enlarged_dims[d];

      numengridpts *= enlarged_dims[d];
    }

    // Allocate the memory for discretized grid count matrix and initialize 
    // it.
    discretized = (double *) malloc(numengridpts * sizeof(double));
    *numenlargedgridpts = numengridpts;
    *gridbinvolume = gvolume;
    gvolume = 1.0 / gvolume;
    for(d = 0; d < numengridpts; d++)
      discretized[d] = 0.0;

    // Now loop over each data and calculate the weights at each grid point.
    for(r = 0; r < num_rows; r++) {

      // First locate the bin the data point falls into and identify it by
      // the lower grid coordinates.
      for(d = 0; d < dim; d++) {
	minindices[d] = floor((dataset[r * dim + d] - mincoords[d])/
			      gridsizes[d]);
      }

      // Assign the weights around the neighboring grid points due to this
      // data point. This results in 2^num_dims number of recursion per data
      // point.
      assignWeights(dataset + r * dim, gridsizes, mincoords, minindices,
		    enlarged_dims, discretized, dim - 1, gvolume, 0, 1);
    }
    free(minindices);
    return discretized;
  }
  
  void gaussify(double *gridsizes, int *enlarged_dims, double *kernelweights,
		int *kernelweights_dims, double acc, double precalc, 
		int level, int pos, int skip) {

    if(level == -1) {
      kernelweights[pos] = exp(precalc * acc);
    }
    else {
      int half = kernelweights_dims[level];
      int g;
      for(g = 0; g <= half; g++) {
	double addThis = g * gridsizes[level];
	double newacc = acc + addThis * addThis;
	int newskip = skip * enlarged_dims[level];
	
	gaussify(gridsizes, enlarged_dims, kernelweights, kernelweights_dims,
		 newacc, precalc, level - 1, pos + skip * g, newskip);
	
	// If this is not the 0th frequency, then do the mirror image thingie.
	if(g != 0) {
	  gaussify(gridsizes, enlarged_dims, kernelweights, 
		   kernelweights_dims, newacc, precalc, level - 1,
		   pos + skip * (enlarged_dims[level] - g), newskip);
	}
      }
    }
  }
  
  
 public:
  
  struct complex {
    double real;
    double imag;
  };

  FFTKde() {}
  
  ~FFTKde() {}

};

#endif
