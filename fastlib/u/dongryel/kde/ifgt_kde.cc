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


//-------------------------------------------------------------------
// File    : ImprovedFastGaussTransform.cpp
// Purpose : Implementation for the Improved Fast Gauss Transform 
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : July 15 2005
//-------------------------------------------------------------------

#include "ifgt_kde.h"
#include "fastlib/fastlib_int.h"
#include <math.h>
#include <values.h>

//-------------------------------------------------------------------
// Constructor 
//
// PURPOSE                                                    
// -------   
// Initialize the class. 
// Read the parameters.
// Allocate memory.
//-------------------------------------------------------------------

ImprovedFastGaussTransform::ImprovedFastGaussTransform(int Dim,
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
						       )		
{	

  //Read the parameters

  d=Dim;
  N=NSources;
  M=MTargets;
  px=pSources;
  h=Bandwidth;
  pq=pWeights;
  py=pTargets;
  p_max=MaxTruncNumber;
  K=NumClusters;
  pci=pClusterIndex;
  pcc=pClusterCenter;
  pcr=pClusterRadii;
  r=CutoffRadius;
  pG=pGaussTransform;
  pT=pTruncNumber;
  eps=epsilon;

  //Memory allocation

  p_max_total=nchoosek(p_max-1+d,d);
  constant_series=new double[p_max_total];
  source_center_monomials = new double[p_max_total];
  target_center_monomials = new double[p_max_total];
  dx = new double[d];
  dy = new double[d];
  heads = new int[d];
  C=new double[K*p_max_total];
	
  h_square=h*h;

  ry=new double[K];
  ry_square=new double[K];
  for(int i=0; i<K; i++)
    {
      ry[i]=r+pcr[i];
      ry_square[i]=ry[i]*ry[i];
    }
		
}

//-------------------------------------------------------------------
// Destructor
//-------------------------------------------------------------------

ImprovedFastGaussTransform::~ImprovedFastGaussTransform()
{
  delete []ry_square;
  delete []ry;
  delete []C;
  delete []heads;
  delete []dy;
  delete []dx;
  delete []target_center_monomials;
  delete []source_center_monomials;
  delete []constant_series;
	
}


//-------------------------------------------------------------------
// Compute the combinatorial number nchoosek.
//-------------------------------------------------------------------

int
ImprovedFastGaussTransform::nchoosek(int n, int k){
  int n_k = n - k;
	
  if (k < n_k)
    {
      k = n_k;
      n_k = n - k;
    }

  int  nchsk = 1; 
  for ( int i = 1; i <= n_k; i++)
    {
      nchsk *= (++k);
      nchsk /= i;
    }

  return nchsk;
}

//-------------------------------------------------------------------
//Computes p_i such error(a,p_i,h) <= q_i epsilon.
//-------------------------------------------------------------------

int
ImprovedFastGaussTransform::return_p(double a_square, int cluster_index)
{
  double a=sqrt(a_square);
  double b,c;
  double error=1;
  double temp=1;
  int p=1;

  while((error > eps) & (p <= p_max))
    {
      b=min(((a+sqrt((a_square)+(2*p*h_square)))/2),ry[cluster_index]);
      c=a-b;
      temp=temp*(((2*a*b)/h_square)/p);
      error=temp*(exp(-(c*c)/h_square));
      p++;
    }

  return p-1;
	
}



//-------------------------------------------------------------------
// This function computes the constants  2^alpha/alpha!.
//-------------------------------------------------------------------
void
ImprovedFastGaussTransform::compute_constant_series(){
	
  int *heads = new int[d+1];
  int *cinds = new int[p_max_total];
	
  for (int i = 0; i < d; i++)
    heads[i] = 0;
  heads[d] = MAXINT;
	
  cinds[0] = 0;
  constant_series[0] = 1.0;
  for (int k=1, t=1, tail=1; k < p_max; k++, tail=t)
    {
      for (int i = 0; i < d; i++)
	{
	  int head = heads[i];
	  heads[i] = t;
	  for ( int j = head; j < tail; j++, t++)
	    {
	      cinds[t] = (j < heads[i+1])? cinds[j] + 1 : 1;
	      constant_series[t] = 2.0 * constant_series[j];
	      constant_series[t] /= (double) cinds[t];
	    }
	}
    }
	
  delete []cinds;
  delete []heads;
	
}

//-------------------------------------------------------------------
// This function computes the monomials [(x_i-c_k)/h]^{alpha}
// and norm([(x_i-c_k)/h])^2
//-------------------------------------------------------------------
void
ImprovedFastGaussTransform::compute_source_center_monomials(int p)
{		

  for (int i = 0; i < d; i++){
    dx[i]=dx[i]/h;
    heads[i] = 0;
  }
		
  source_center_monomials[0] = 1.0;
  for (int k=1, t=1, tail=1; k < p; k++, tail=t){
    for (int i = 0; i < d; i++){
      int head = heads[i];
      heads[i] = t;
      for ( int j = head; j < tail; j++, t++)
	source_center_monomials[t] = dx[i] * source_center_monomials[j];
    }						
  }					

}

//-------------------------------------------------------------------
// This function computes the monomials [(y_j-c_k)/h]^{alpha}
//-------------------------------------------------------------------
void
ImprovedFastGaussTransform::compute_target_center_monomials()
{		

  for (int i = 0; i < d; i++){
    dy[i]=dy[i]/h;
    heads[i] = 0;
  }
		
  target_center_monomials[0] = 1.0;
  for (int k=1, t=1, tail=1; k < p_max; k++, tail=t){
    for (int i = 0; i < d; i++){
      int head = heads[i];
      heads[i] = t;
      for ( int j = head; j < tail; j++, t++)
	target_center_monomials[t] = dy[i] * target_center_monomials[j];
    }						
  }					

}

//-------------------------------------------------------------------
// This function computes the coeffeicients C_k for all clusters.
//-------------------------------------------------------------------
void
ImprovedFastGaussTransform::compute_C()
{

  for (int i = 0; i < K*p_max_total; i++){
    C[i]=0.0;
  }

  p_max_actual=-1;

  for(int i=0; i<N; i++){
    int k=pci[i];

    int source_base=i*d;
    int center_base=k*d;

    source_center_distance_square=0.0;

    for (int j = 0; j < d; j++){
      dx[j]=(px[source_base+j]-pcc[center_base+j]);
      source_center_distance_square += (dx[j]*dx[j]);
    }

    pT[i]=return_p(source_center_distance_square,k);

    if (pT[i]>p_max_actual){
      p_max_actual=pT[i];
    }
	
    compute_source_center_monomials(pT[i]);		
		
    double f=pq[i]*exp(-source_center_distance_square/h_square);

    for(int alpha=0; alpha<nchoosek(pT[i]-1+d,d); alpha++){
      C[k*p_max_total+alpha]+=(f*source_center_monomials[alpha]);
    }
		
  }

  p_max_actual_total=nchoosek(p_max_actual-1+d,d);

  compute_constant_series();
	
  for(int k=0; k<K; k++){
    for(int alpha=0; alpha<p_max_total; alpha++){
      C[k*p_max_total+alpha]*=constant_series[alpha];
    }
  }


}

//-------------------------------------------------------------------
// Actual function to evaluate the Gauss Transform.
//-------------------------------------------------------------------

void
ImprovedFastGaussTransform::Evaluate()
{
	
  compute_C();	

  for(int j=0; j < M; j++)
    {
      pG[j]=0.0;	

      int target_base=j*d;	    	
		
      for(int k=0; k<K; k++){

	int center_base=k*d;

	double  target_center_distance_square=0.0;
	for(int i=0; i<d; i++){
	  dy[i]=py[target_base+i]-pcc[center_base+i];
	  target_center_distance_square += dy[i]*dy[i];
	  if (target_center_distance_square > ry_square[k]) break;
	}

	if (target_center_distance_square <= ry_square[k]){
	  compute_target_center_monomials();
	  double g=exp(-target_center_distance_square/h_square);
	  for(int alpha=0; alpha<p_max_actual_total; alpha++){
	    pG[j]+=(C[k*p_max_total+alpha]*g*target_center_monomials[alpha]);
	  }											
	}
      }
    }
	

}

