/*
 *  angle_matcher.h
 *  
 *  Computes three point correlations for a matcher specified by a range of 
 *  values of r1, a factor c such that r2 = c*r1, and a range of angles between 
 *  the sides theta.
 *
 *  Created by William March on 7/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ANGLE_MATCHER_H
#define ANGLE_MATCHER_H

#include "fastlib/fastlib.h"


// Assumptions (for now):
//
// bins might overlap (especially at large values of theta)
// Values of r1 are spaced far enough apart such that a tuple of points will 
// only satisfy one

// IMPORTANT: I think I'm assuming that r2 is enough larger than r1 that there
// isn't any overlap

namespace npt {

  class AngleMatcher {
    
  private:
    
    std::vector<double> short_sides_;
    // the long side is this times the short side
    // should I include more than one of these?
    double long_side_multiplier_;
    
    std::vector<double> long_sides_;

    // these are in radians
    std::vector<double> thetas_;
    
    // indexed by [value of r1][value of theta]
    std::vector<std::vector<double> > r3_sides_;
    
    // this is the value of theta where r2 = r3
    // computed by arccos(1/2k) where k is long_side_multiplier_
    double theta_cutoff_;
    // this is the index in the theta array where it occurs
    // i.e. thetas_[theta_cutoff_index_] is the first theta where r3 > r2
    int theta_cutoff_index_;
    
    double cos_theta_cutoff_;
    
    // upper and lower bound arrays
    // these include the half bandwidth added or subtracted
    std::vector<double> r1_lower_sqr_;
    std::vector<double> r1_upper_sqr_;
    
    std::vector<double> r2_lower_sqr_;
    std::vector<double> r2_upper_sqr_;
    
    // these are indexed by r1 value, then by angle/r3
    std::vector<std::vector<double> > r3_lower_sqr_;
    std::vector<std::vector<double> > r3_upper_sqr_;
    
    
    
    
    // This is 0.25 in the thesis (or maybe 0.1?)
    // Note that the bin thickness is this times the scale
    // Does this mean in each dimension?
    double bin_thickness_factor_;
    
    double min_area_sq_;
    double max_area_sq_;
    
    double longest_possible_side_sqr_;
    double shortest_possible_side_sqr_;
    
    /*
    int num_min_area_prunes_;
    int num_max_area_prunes_;
    
    int num_r1_prunes_;
    int num_r3_prunes_;
    */
    int num_large_r1_prunes_, num_small_r1_prunes_;
    int num_large_r3_prunes_;
    int num_large_r2_prunes_, num_small_r2_prunes_;
     
    ////////////////////////////
    
    double ComputeR3_(double r1, double r2, double theta);
    
    
  public:
    
    AngleMatcher(std::vector<double>& short_sides, double long_side,
                 std::vector<double>& thetas, double bin_size) :
    short_sides_(short_sides), long_side_multiplier_(long_side), 
    thetas_(thetas), bin_thickness_factor_(bin_size),
    r3_sides_(short_sides.size()), long_sides_(short_sides.size()),
    r1_lower_sqr_(short_sides.size()), r1_upper_sqr_(short_sides.size()), 
    r2_lower_sqr_(short_sides.size()), r2_upper_sqr_(short_sides.size()),
    r3_lower_sqr_(short_sides.size()), r3_upper_sqr_(short_sides.size()) {
      

      cos_theta_cutoff_ = 1.0 / (2.0 * long_side_multiplier_);
      theta_cutoff_ = acos(cos_theta_cutoff_);
      theta_cutoff_index_ = std::lower_bound(thetas_.begin(), thetas_.end(), 
                                             theta_cutoff_) - thetas_.begin();
      
      
      
      double half_thickness = bin_thickness_factor_ / 2.0;

      
      for (int i = 0; i < short_sides_.size(); i++) {
          
        r3_sides_[i].resize(thetas_.size());
        r3_lower_sqr_[i].resize(thetas_.size());
        r3_upper_sqr_[i].resize(thetas_.size());
        
        long_sides_[i] = long_side_multiplier_ * short_sides_[i];
        
        r1_lower_sqr_[i] = ((1.0 - half_thickness) * short_sides_[i])
                            * ((1.0 - half_thickness) * short_sides_[i]);
        r1_upper_sqr_[i] = ((1.0 + half_thickness) * short_sides_[i])
                            * ((1.0 + half_thickness) * short_sides_[i]);
        
        r2_lower_sqr_[i] = ((1.0 - half_thickness) * long_sides_[i])
        * ((1.0 - half_thickness) * long_sides_[i]);
        r2_upper_sqr_[i] = ((1.0 + half_thickness) * long_sides_[i])
        * ((1.0 + half_thickness) * long_sides_[i]);
        
        for (int j = 0; j < thetas_.size(); j++) {
          
          r3_sides_[i][j] = ComputeR3_(short_sides_[i], 
                                     long_sides_[i], 
                                     thetas_[j]);
          
          r3_lower_sqr_[i][j] = ((1.0 - half_thickness) * r3_sides_[i][j])
                              * ((1.0 - half_thickness) * r3_sides_[i][j]);
          r3_upper_sqr_[i][j] = ((1.0 + half_thickness) * r3_sides_[i][j])
                              * ((1.0 + half_thickness) * r3_sides_[i][j]);
          
        } // for j
        
      } // for i
      
      double half_min_perimeter = (sqrt(r1_lower_sqr_[0]) 
                                   + sqrt(r2_lower_sqr_[0]) 
                                   + sqrt(r3_lower_sqr_[0][0])) / 2.0;
      double half_max_perimeter = (sqrt(r1_upper_sqr_.back()) 
                                   + sqrt(r2_upper_sqr_.back()) 
                                   + sqrt(r3_upper_sqr_.back().back())) / 2.0;
      min_area_sq_ = half_min_perimeter 
      * (half_min_perimeter - sqrt(r1_lower_sqr_[0])) 
            * (half_min_perimeter - sqrt(r2_lower_sqr_[0])) 
            * (half_min_perimeter - sqrt(r3_lower_sqr_[0][0]));
      max_area_sq_ = half_max_perimeter 
      * (half_max_perimeter - sqrt(r1_upper_sqr_.back())) 
      * (half_max_perimeter - sqrt(r2_upper_sqr_.back())) 
      * (half_max_perimeter - sqrt(r3_upper_sqr_.back().back()));
      
      /*
      num_min_area_prunes_ = 0;
      num_max_area_prunes_ = 0;
      num_r1_prunes_ = 0;
      num_r3_prunes_ = 0;
      */
      
      longest_possible_side_sqr_ = max(r2_upper_sqr_.back(), 
                                       r3_upper_sqr_.back().back());
      shortest_possible_side_sqr_ = min(r1_lower_sqr_.front(), 
                                        r3_lower_sqr_.front().front());
      
      num_large_r1_prunes_ = 0;
      num_small_r1_prunes_ = 0;
      num_large_r3_prunes_ = 0;
      num_large_r2_prunes_ = 0;
      num_small_r2_prunes_ = 0;
      
      // IMPORTANT: I'm not sure the upper and lower sqr arrays are still sorted
      // especially for r3
      
      
      mlpack::IO::Info << "r1_lower_sqr_: ";
      for (int i = 0; i < r1_lower_sqr_.size(); i++) {
        
        mlpack::IO::Info << r1_lower_sqr_[i] << ", ";
        
      }
      mlpack::IO::Info << "\n";

      mlpack::IO::Info << "r1_upper_sqr_: ";
      for (int i = 0; i < r1_upper_sqr_.size(); i++) {
        
        mlpack::IO::Info << r1_upper_sqr_[i] << ", ";
        
      }
      mlpack::IO::Info << "\n";
      
      
      mlpack::IO::Info << "r2_lower_sqr_: ";
      for (int i = 0; i < r2_lower_sqr_.size(); i++) {
        
        mlpack::IO::Info << r2_lower_sqr_[i] << ", ";
        
      }
      mlpack::IO::Info << "\n";
      
      mlpack::IO::Info << "r2_upper_sqr_: ";
      for (int i = 0; i < r2_upper_sqr_.size(); i++) {
        
        mlpack::IO::Info << r2_upper_sqr_[i] << ", ";
        
      }
      mlpack::IO::Info << "\n";
      
      mlpack::IO::Info << "r3_lower_sqr_: ";
      for (int i = 0; i < r3_lower_sqr_[0].size(); i++) {
        
        mlpack::IO::Info << r3_lower_sqr_[0][i] << ", ";
        
      }
      mlpack::IO::Info << "\n";
      
      mlpack::IO::Info << "r3_upper_sqr_: ";
      for (int i = 0; i < r3_upper_sqr_[0].size(); i++) {
        
        mlpack::IO::Info << r3_upper_sqr_[0][i] << ", ";
        
      }
      mlpack::IO::Info << "\n";
      
      
      mlpack::IO::Info << "longest side: " << longest_possible_side_sqr_ << "\n";
      mlpack::IO::Info << "shortest side: " << shortest_possible_side_sqr_ << "\n";
      
    } // constructor
    
    // returns the index of the value of r1 that is satisfied by the tuple
    // the list contains the indices of thetas_ that are satisfied by the tuple
    // assumes that valid_theta_indices is initialized and empty
    int TestPointTuple(arma::colvec& vec1, arma::colvec& vec2, 
                       arma::colvec& vec3,
                       std::vector<int>& valid_theta_indices);
    
    // returns true if the tuple of nodes might contain a tuple of points that
    // satisfy one of the matchers
    // If false, then pruning is ok
    bool TestNodeTuple(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                       const DHrectBound<2>& box3);
    
    
    void PrintNumPrunes() {
     
      /*
      mlpack::IO::Info << "Num r1 prunes: " << num_r1_prunes_ << "\n";
      mlpack::IO::Info << "Num r3 prunes: " << num_r3_prunes_ << "\n";
      mlpack::IO::Info << "Num min area prunes: " << num_min_area_prunes_ << "\n";
      mlpack::IO::Info << "Num max area prunes: " << num_max_area_prunes_ << "\n";
      */
      
      mlpack::IO::Info << "Num large r1 prunes: " << num_large_r1_prunes_ << "\n";
      mlpack::IO::Info << "Num small r1 prunes: " << num_small_r1_prunes_ << "\n";
      mlpack::IO::Info << "Num large r3 prunes: " << num_large_r3_prunes_ << "\n";
      mlpack::IO::Info << "Num large r2 prunes: " << num_large_r2_prunes_ << "\n";
      mlpack::IO::Info << "Num small r2 prunes: " << num_small_r2_prunes_ << "\n";
      
    }
    
    
  }; // AngleMatcher

}


#endif