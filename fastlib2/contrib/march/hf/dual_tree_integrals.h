/**
 * @file dual_tree_integrals.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Code for computing two electron integrals with a dual-tree approximation
 */

#ifndef DUAL_TREE_INTEGRALS_H
#define DUAL_TREE_INTEGRALS_H

#include <fastlib/fastlib.h>

/**
 * Algorithm class for computing two-electron integrals.
 */
class DualTreeIntegrals {
  
  FORBID_ACCIDENTAL_COPIES(DualTreeIntegrals);
  
  friend class DualTreeIntegralsTest;
  
  DualTreeIntegrals() {}
  
  ~DualTreeIntegrals() {}
  
  /**
   * Stat class for tree building.  
   *
   * I'm not sure I actually need one right now. 
   */
  class IntegralStat {
   
   private:
   
    
    
   public:
    void Init() {
      
    } // Init
    
    void Init(const Matrix& matrix, index_t start, index_t count) {
      
    } // Init (leaves)
    
    void Init(const Matrix& matrix, index_t start, index_t count, 
              const IntegralStat& left, const IntegralStat& right) {
    
      
      
    } // Init (non-leaves)
    
  }; // class IntegralStat
  
  // This assumes identical bandwidth small Gaussians
  // Otherwise, I'll need something other than a Matrix
  // I should also consider something better than bounding boxes
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, IntegralStat> IntegralTree; 
  
  
private:
    
  // The tree 
  IntegralTree* tree_;
  
  // The centers of the identical, spherical Gaussian basis functions
  Matrix centers_;
  
  // The fx module
  struct datanode* module_;
  
  // The common bandwidth of all the basis functions
  double bandwidth_;
  
    
public:
  
  /**
   * Initialize the class with the centers of the data points, the fx module,
   * bandwidth
   */
  void Init(const Matrix& centers_in, struct datanode* mod, double band) {
  
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    bandwidth_ = band;
    
  }
  
  
  
  
  
}; // class DualTreeIntegrals






#endif 