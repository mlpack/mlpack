/**
 * @file hybrid_error_stat.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Defines the stat classes for the different kinds of error
 */
 
#ifndef HYBRID_ERROR_STAT_H
#define HYBRID_ERROR_STAT_H

// I should make all of these inherit the basic stat stuff
// The classes should be able to overlap entirely except for the Epsilon_ 
// function and some private variables for the complicated functions 

class GenericErrorStat {

 protected:
 
  index_t remaining_references_;
  
  // This isn't right as a global max error bound, since it should apply to each
  // query separately
  double error_incurred_;
  
 // double query_upper_bound_;
  double query_lower_bound_;
  
  virtual double Epsilon_() = 0;
  
public:
    
  GenericErrorStat() {}
    
  virtual ~GenericErrorStat() {}
    
  void Init(const Matrix& matrix, index_t start, index_t count) {
      
    //query_count_ = -1;
    error_incurred_ = 0.0;
      
  } // Init() (leaves)
  
  void Init(const Matrix& matrix, index_t start, index_t count, 
            const GenericErrorStat& left, const GenericErrorStat& right) {
    
    //query_count_ = -1;
    error_incurred_ = 0.0;
    
  } // Init() (non-leaves)
  
  bool CanPrune(double q_upper_bound, double q_lower_bound, 
                index_t reference_count) {
    
    bool prune = false;
    
    double max_error_incurred = 0.5 * (q_upper_bound - q_lower_bound);
    DEBUG_ASSERT(max_error_incurred >= 0.0);
    
    double allowed_error = Epsilon_() * 
      reference_count / remaining_references_;
    
    
    
    DEBUG_ASSERT(allowed_error >= 0.0); 
    
  //  printf("allowed_error = %g\n", allowed_error);
     
    if (max_error_incurred < allowed_error) {
      
      prune = true;
      
      //error_incurred_ = error_incurred_ + max_error_incurred;
      
      //remaining_references_ = remaining_references_ - reference_count;
      DEBUG_ASSERT(remaining_references_ >= 0);
      
    }
    
    return prune;
    
  } // CanPrune()  
  
  void set_remaining_references(index_t new_count) {
  
    remaining_references_ = new_count;
    
    DEBUG_ASSERT(remaining_references_ >= 0);
  
  } // set_query_count()
  
  index_t remaining_references() {
  
    return remaining_references_;
  
  } // query_count()
  
  /*void set_query_upper_bound(double bd) {
    query_upper_bound_ = bd;
  }
  
  double query_upper_bound() {
    return query_upper_bound_;
  }
  */
  
  void set_query_lower_bound(double bd) {
    query_lower_bound_ = bd;
  }
  
  double query_lower_bound() {
    return query_lower_bound_;
  }

}; // GenericErrorStat

/**
 * Prunes with an absolute error criterion
 */
class AbsoluteErrorStat : public GenericErrorStat {

 private: 

  double max_error_;

 protected:

  /**
   * Returns the error tolerance as a function of the bounds on Q.  In this case
   * we divide by the lower bound to get absolute error.
   */ 
  double Epsilon_() {
  
    double eps = max_error_;
    DEBUG_ASSERT(eps >= 0.0);
    
    return (eps);
  
  } // Epsilon_
  

 public:
 
  AbsoluteErrorStat() {}
    
  ~AbsoluteErrorStat() {}
  
  void SetParams(double max_err, double min_err, double steep) {
  
    max_error_ = max_err;
  
  } // SetParams()

}; // class AbsoluteErrorStat


/** 
 * Prunes with a relative error criterion
 */
class RelativeErrorStat : public GenericErrorStat {

 private:

  double max_error_;

protected:
  /**
   * Relative error just depends on epsilon_
   */
  double Epsilon_() {
      
    double eps = max_error_ * query_lower_bound_;
    DEBUG_ASSERT(eps >= 0.0);
    return eps;
    
  } // Epsilon_

 public:
 
  RelativeErrorStat() {}
    
  ~RelativeErrorStat() {}
    
  void SetParams(double max_err, double min_err, double steep) {
  
    max_error_ = max_err;
  
  } // SetParams  
  
  
}; // class RelativeErrorStat


/**
 * Prunes with the hybrid exponential error criterion 
 */
class ExponentialErrorStat : public GenericErrorStat {

 private:
  
  double max_error_;
  
  double steepness_;
  
  double min_error_;
  
  
protected:
  /**
   * Hybrid error using the exponential criterion
   */
  double Epsilon_() {
    
    double eps = (max_error_ * exp(-steepness_ * query_lower_bound_)) + 
        min_error_;
        
    eps = eps * query_lower_bound_;
    
    DEBUG_ASSERT(eps >= 0.0);
    
    return (eps);
    
  } // Epsilon_()
  
 public:
 
  ExponentialErrorStat() {}
    
  ~ExponentialErrorStat() {}

    void SetParams(double max_err, double min_err, double steep) {
    
      max_error_ = max_err;
    
      steepness_ = steep;
      
      min_error_ = min_err;
      
      error_incurred_ = 0.0;
         
    } // SetParams()
  
  
}; // class ExponentialErrorStat

/**
 * Uses a Gaussian hybrid error criterion
 */
class GaussianErrorStat : public GenericErrorStat {

private:
  
  double max_error_;
  
  double steepness_;
  
  double min_error_;
  
  
protected:
  /**
    * Hybrid error using the gaussian criterion
   */
  double Epsilon_() {
    
    double eps = (max_error_ * 
                  exp(-steepness_ * query_lower_bound_ * query_lower_bound_)) 
                  + min_error_ + error_incurred_;
                  
    eps = eps * query_lower_bound_;
    
    DEBUG_ASSERT(eps >= 0.0);

    return (eps);
    
  } // Epsilon_()
  
 public:

  GaussianErrorStat() {}
    
  ~GaussianErrorStat() {}
    
    void SetParams(double max_err, double min_err, double steep) {
    
      max_error_ = max_err;
      
      steepness_ = steep;
      
      min_error_ = min_err;
      
      error_incurred_ = 0.0;
      
    } // SetParams  

}; // class GaussianErrorStat

class HybridErrorStat : public GenericErrorStat {

private:

  double steepness_;
  
  double max_error_;
  
  double min_error_;
  
  
protected:

  double Epsilon_() {
  
    // Would like to multiply this by 1 - exp(query_upper_bound)
    double eps = min_error_ * query_lower_bound_;
    
    eps = eps + 
        max_error_ * exp(-steepness_ * query_lower_bound_);
    
    
    DEBUG_ASSERT(eps >= 0.0);
    
    return eps;
  
  } // Epsilon_()

public:

    HybridErrorStat() {}
    
  ~HybridErrorStat() {}
  
  void SetParams(double max_err, double min_err, double steep) {
  
    steepness_ = steep;
    
    max_error_ = max_err;
    
    min_error_ = min_err;
  
  } // SetParams()

};  // class HybridErrorStat



#endif