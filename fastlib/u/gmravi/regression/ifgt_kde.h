//-------------------------------------------------------------------
// The code was written by Vikas Raykar and Changjiang Yang 
// and is copyrighted under the Lesser GPL: 
//
// Copyright (C) 2006 Vikas Raykar and Changjiang Yang 
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; version 2.1 or later.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU Lesser General Public License for more details. 
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, 
// MA 02111-1307, USA.  
//
// The author may be contacted via email at:
// vikas(at)umiacs(.)umd(.)edu, cyang(at)sarnoff(.)com
//-------------------------------------------------------------------


//-------------------------------------------------------------
// File    : ImprovedFastGaussTransform.h
// Purpose : Interface for 
//			 Data Adaptive Improved Fast Gauss Transform.
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : July 15 2005
//-------------------------------------------------------------
// Data Adaptive Improved Fast Gauss Transform (IFGT).
// All Sources have the same scales 'h'.
//
// 
// A new version of the IFGT where the parameters are chosen
// based on the acutal distribution of the source points.
// The truncation number for each source point is chosen based
// on its distance to the cluster center.
//
// Advantages:
// -----------
// 1. Better Speedup.  
// 2. Choice of parameters is fully automatic taking into
//    consideration the actual distribtion of the data points.
// 3. Uses more tight pointwise error bounds in choosing the
//    parameters.
//
// Implementation based on:
//
// Fast computation of sums of Gaussians in high dimensions. 
// Vikas C. Raykar, C. Yang, R. Duraiswami, and N. Gumerov,
// CS-TR-4767, Department of computer science,
// University of Maryland, Collegepark.
// ------------------------------------------------------------


#ifndef IMPROVED_FAST_GAUSS_TRANSFORM_H
#define IMPROVED_FAST_GAUSS_TRANSFORM_H

class ImprovedFastGaussTransform{
 public:
  //constructor 
  ImprovedFastGaussTransform(int Dim,
			     int NSources,
			     int MTargets,
			     double *pSources,
			     double Bandwidth,
			     double *pWeights,
			     double *pTargets,
			     int MaxTruncNumber,
			     int NumClusters,
			     int *pClusterIndex, 
			     double *pClusterCenter,
			     double *pClusterRadii,
			     double CutoffRadius,
			     double epsilon,
			     double *pGaussTransform,
			     int *pTruncNumber
			     );		

  //destructor
  ~ImprovedFastGaussTransform();				

  //function to evaluate the Gauss Transform.
  void Evaluate();

 private:
  //Parameters

  int d;				
  int N;				
  int M;				
  double *px;			
  double  h;			
  double *pq;			
  double *py;	
  int p_max;	
  int K;	
  int *pci; 
  double *pcc;  
  double *pcr;
  double r;
  double eps;


  double *pG;         
  int *pT;			       		      

  //

  int     p_max_total;
  int     p_max_actual;
  int     p_max_actual_total;
  double *constant_series;
  double *source_center_monomials;
  double  source_center_distance_square;
  double *target_center_monomials;
  double  target_center_distance_square;
  double *dx;
  double *dy;
  int	   *heads;		
  double *C;
  double h_square;
  double *ry;
  double *ry_square;

  //Functions

  int  nchoosek(int n, int k);
  int  return_p(double a_square, int cluster_index);
  void compute_constant_series();
  void compute_source_center_monomials(int p);
  void compute_target_center_monomials();
  void compute_C();
    
};


#endif
