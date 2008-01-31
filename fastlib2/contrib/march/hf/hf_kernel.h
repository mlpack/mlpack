#ifndef HF_KERNEL_H
#define HF_KERNEL_H

/** 
* @file hf_kernel.h
*
* Kernels for the Hartree-Fock molecular orbital calculation
*
*/


/** 
* Evaluates the kernel given four centers.
*
* Note: currently assuming a global fixed bandwidth
*/
class HFFourBodyKernel {
  
private:
  
  HFKernel hf_kernel_;
  
  //I believe this is the 4x4 distances matrix;
  Matrix distmat_;
  
public:
  
  /**
  * Initializes to a specific bandwidth
  *
  * @param bandwidth_in the given bandwidth
  */
  void Init(double bandwidth_in) {
   
    hf_kernel_.Init(bandwidth_in);
    distmat_.Init(3,3);
    
  }
  
  //Need to figure out what this is, exactly . . .
  int order() {
      return 4;
  }
  
  /**
  * Evaluates the potential on a given set of distances
  *
  * Note: why does this need to be a matrix, it only depends on 3 distances
  *
  * @param sqdists the 4x4 matrix of distances i, j, k, l
  */
  double EvalUnnormOnSq(const Matrix &sqdists) const {
    
    double result = 1;
    // compute the two Gaussian kernel distances
    // then, compute the weird function on the average distance 
    
    
    return result;
    
  }
  
  
}


class HFKernel {
  
  //Does Dong's code create one of these for every bounds computation?
  // Seems inefficient to do so, can just reuse the same one with different distances
  
private:
  
  GaussianKernel gauss_kernel_;
  
  double bandwidth;
  
  
public:
  
  void Init(double bandwidth_in) {
    bandwidth = bandwidth_in;
    gauss_kernel_.Init(bandwidth_in);
  }
  
  double EvalUnnormOnSq(double sqdist) const {
   
    //Need to return the full computation here
    double total;
    
    return(total);
    
  }
  
  
}






#endif 