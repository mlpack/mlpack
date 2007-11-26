/*
 * =====================================================================================
 *
 *       Filename:  spe.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/20/2007 10:29:44 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <math.h>
#include <limits>
#include "spe.h"

void SPE::Init(std::string filename) {
  data_.InitFromFile(filename.c_str());
	num_of_points_=data_.matrix().n_cols();
	old_dimension_=data_.matrix().n_rows();
  lambdas_=NULL;
	set_lambdas(2, 0.1, 100);
  set_new_dimensions(1);
	set_hits_per_point(1);
	tolerance_=0.1;
}
void SPE::Destruct() {
	if (lambdas_!=NULL) {
    delete []lambdas_;	
	}
}
void SPE::Optimize(float32 range, std::string out_file, 
		               std::vector<float32> &stress){
  Matrix newdims;
	newdims.Init(new_dimension_, num_of_points_);
	// initialize new dimensions
	for(index_t i=0; i<newdims.n_rows(); i++) {
	  for(index_t j=0; j<newdims.n_cols(); j++) {
		  newdims.set(i, j, 1.0 * rand() / RAND_MAX);
		}
	}
  //now run the actual algorithm
	stress.push_back(std::numeric_limits<float32>::max());
  for(index_t i=0; i<num_of_lambdas_; i++) {
    NONFATAL("Now processing lambda: %lg", lambdas_[i]);
    float32 stress_err=1;		
    while (stress_err>tolerance_) {
		  stress.push_back(0);
		  for(index_t j=0; j<1000; j++) {
		    for(index_t k=0; k<num_of_points_; k++){
			    index_t neighbor= rand() % num_of_points_;
				  float32 r_ij=Distance(data_.matrix().GetColumnPtr(k),
					                      data_.matrix().GetColumnPtr(neighbor), 
						  									old_dimension_);
				  float32 d_ij=Distance(newdims.GetColumnPtr(k), 
			                 newdims.GetColumnPtr(neighbor), new_dimension_);
				
				  if ((k!=neighbor) && (r_ij<range || d_ij<r_ij) ) {
				    float32 temp;
					  float32 adjustment=lambdas_[i]/2*(r_ij-d_ij)/(d_ij+1e-10);
					  for(index_t l=0;l<new_dimension_; l++) {
					    temp=adjustment * (newdims.GetColumnPtr(k)[l]-
							     newdims.GetColumnPtr(neighbor)[l]);
					    newdims.GetColumnPtr(k)[l] +=temp;
              newdims.GetColumnPtr(neighbor)[l]-=temp;
					  }
				    stress.back() +=(r_ij-d_ij)*(r_ij-d_ij)/r_ij; 
				  }     
	  		}	
			}
      stress_err=fabs(stress[stress.size()-1]- 
						        stress[stress.size()-2])/stress[stress.size()-1];

		  NONFATAL("..... Total Stress: %lg, stress_err: %lg\n",
				       	stress.back(), stress_err); 
	  }
	}
	data::Save(out_file.c_str(), newdims);
}

