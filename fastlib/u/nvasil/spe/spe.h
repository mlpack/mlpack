/*
 * =====================================================================================
 * 
 *       Filename:  spe.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  11/20/2007 10:42:30 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef SPE_H_
#define SPE_H_

#include <string>
#include <vector>
#include "fastlib/fastlib.h"

class SPE {
 public:
	 void Init(std::string filename);
	 void Destruct();
	 void Optimize(float32 range, std::string out_file, 
			           std::vector<float32> &stress);
	 void set_hits_per_point(index_t hits_per_point) {
	   hits_per_point_ = hits_per_point;
	 }
	 void set_new_dimensions(index_t new_dimension) {
	   new_dimension_=new_dimension;
	 }
	 void set_lambdas(float32 lambda_max, float32 lambda_min, 
			              index_t num_of_lambdas) {
		 num_of_lambdas_=num_of_lambdas;
	   if (lambdas_!=NULL) {
		   delete []lambdas_;
		 }
		 lambdas_=new float32[num_of_lambdas];
		 float32 delta_lambda= (lambda_max-lambda_min)/num_of_lambdas_;
		 for(index_t i=0; i<num_of_lambdas_; i++) {
		   lambdas_[i]=lambda_max-i*delta_lambda; 
		 }
	 }
	 void set_tolerance(float32 tolerance) {
	   tolerance_=tolerance;
	 }
 private:
	index_t hits_per_point_;
	float32 *lambdas_;
	index_t num_of_lambdas_;
	index_t old_dimension_;
	index_t new_dimension_;
	index_t num_of_points_;
  Dataset data_;
	float32 tolerance_;
  float32 Distance(double *p1, double *p2, int len) {
	  float32 dist=0;
		for(index_t i=0; i<len; i++) {
		  dist+=(p1[i]-p2[i])*(p1[i]-p2[i]);
		}
		return sqrt(dist);
	};

};

#endif
