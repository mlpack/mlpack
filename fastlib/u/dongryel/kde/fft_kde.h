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

    int b,index1,index2,trans_size,trans;
    double pi2 = 4.*asin(1.);
    
    // used in recursive formula for Re(W^b) and Im(W^b)
    double pi2n,cospi2n,sinpi2n;
    
    // wk = W^k = e^(2 pi i b/N) in the Danielson-Lanczos formula for a
    // transform of length N
    struct complex wb;
    
    // buffers for implementing recursive formulas
    struct complex temp1,temp2;

    // treat f as an array of N complex numbers
    struct complex *c = (struct complex *)f;

    // Place the elements of the array c in bit-reversed order
    for(index1 = 1, index2 = 0; index1 < N; index1++) {
      
      // to find the next bit reversed array index subtract leading 1's from
      // index2
      for(b = N / 2; index2 >= b; b /= 2) {
	index2-=b;
      }
      
      // Next replace the first 0 in index2 with a 1 and this gives the correct
      // next value
      index2+=b;
      
      // swap each pair only the first time it is found
      if(index2 > index1) {
	temp1 = c[index2*skip];
	c[index2*skip] = c[index1*skip];
	c[index1*skip] = temp1;
      }
    }

    // Next perform successive transforms of length 2,4,...,N using the
    // Danielson-Lanczos formula

    // trans_size = size of transform being computed
    for(trans_size=2;trans_size<=N;trans_size*=2) {
      
      // +- 2 pi/trans_size
      pi2n = forward*pi2/(double)trans_size;
      
      // Used to calculate W^k in D-L formula
      cospi2n = cos(pi2n);
      sinpi2n = sin(pi2n);
      
      // Initialize W^b for b=0
      wb.real = 1.;
      wb.imag = 0.;
      
      // Step over half of the elements in the transform
      for(b = 0;b < trans_size / 2; b++) {
	
	// Iterate over all transforms of size trans_size to be computed
	for(trans = 0; trans < N / trans_size; trans++) {

	  // Index of element in first half of transform being computed
	  index1 = (trans*trans_size+b)*skip;

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
      for(index1=0;index1<skip*N;index1+=skip) {
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

    int i,j,dim;

    // These determine where to begin successive transforms and the skip 
    // between their elements (see below)
    int planesize=1,skip=1;

    // Total size of the ndims dimensional array
    int totalsize=1;
    
    // determine total size of array
    for(dim=0;dim<ndims;dim++) {
      totalsize *= size[dim];
    }
    
    // loop over dimensions
    for(dim=ndims-1;dim>=0;dim--) {
      
      // planesize = Product of all sizes up to and including size[dim] 
      planesize *= size[dim];

      // Take big steps to begin loops of transforms 
      for(i=0;i<totalsize;i+=planesize) {
	
	// Skip sets the number of transforms in between big steps as well as 
	// the skip between elements
	for(j=0;j<skip;j++) {

	  // 1-D Fourier transform. (Factor of two converts complex index to 
	  // double index.)
	  fftc1(f+2*(i+j),size[dim],skip,forward);
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
    double pi2n = 4.*asin(1.)/N,cospi2n=cos(pi2n),sinpi2n=sin(pi2n);
    
    // wb = W^b = e^(2 pi i b/N) in the Danielson-Lanczos formula for a 
    // transform of length N
    struct complex wb;
    
    // Buffers for implementing recursive formulas
    struct complex temp1,temp2;
    
    // Treat f as an array of N/2 complex numbers
    struct complex *c = (struct complex *)f;
    
    // Do a transform of f as if it were N/2 complex points
    if(forward==1) {
      fftc1(f,N/2,1,1);
    }

    // initialize W^b for b = 0
    wb.real = 1.;
    wb.imag = 0.;

    // Loop over elements of transform. See documentation for these formulas 
    for(b=1;b<N/4;b++) {

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
    
    if(forward==-1) {
      c[0].real *= .5;
      c[0].imag *= .5;
      fftc1(f,N/2,1,-1);
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
