//-------------------------------------------------------------------
// The code was written by Changjiang Yang and Vikas Raykar
// and is copyrighted under the Lesser GPL: 
//
// Copyright (C) 2006  Changjiang Yang and Vikas Raykar
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
// The author may be contacted via email at:cyang(at)sarnoff(.)com
// vikas(at)umiacs(.)umd(.)edu
//-------------------------------------------------------------------

//-------------------------------------------------------------------
// File    : KCenterClustering.cpp
// Purpose : Implementation for the k-center clustering algorithm.
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : April 25 2005, June 10 2005, August 23, 2005
//-------------------------------------------------------------------

#include "kcenter_clustering.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "fastlib/fastlib_int.h"

//-------------------------------------------------------------------
// Constructor 
//
// PURPOSE                                                    
// -------   
// Initialize the class. 
// Read the parameters.
//
// INPUT                                                      
// ----------
// Dim			   --> dimension of the points.
// NSources		   --> number of sources.
// pSources		   --> pointer to sources, (d*N).
// pClusterIndex   --> pointer to a vector of length N where the 
//                     i th element is the cluster number to 
//					   which the i th point belongs.
//	
//-------------------------------------------------------------------

KCenterClustering::KCenterClustering(int Dim,
			int NSources,
			double *pSources,
			int *pClusterIndex,
			int NumClusters
			)
{	

	//Read the parameters

	d=Dim;
	N=NSources;
	px=pSources;
	pci=pClusterIndex;
	K=NumClusters;
	dist_C = new double[N]; //distances to the center.
	r=new double[K];


}

//-------------------------------------------------------------------
// Destructor
//-------------------------------------------------------------------

KCenterClustering::~KCenterClustering()
{	
	delete [] dist_C;
	delete [] r;
}



//-------------------------------------------------------------------
// ddist is the square of the distance of two vectors(double)
//-------------------------------------------------------------------


double
KCenterClustering::ddist(const int d, const double *x, const double *y)
{
	double t, s = 0.0;
	for (int i = d; i != 0; i--)
	{
		t = *x++ - *y++;
		s += t * t;
	}
	return s;
}



//-------------------------------------------------------------------
// Find the largest element from a vector
//-------------------------------------------------------------------

int
KCenterClustering::idmax(int n, double *x)
{
	int k = 0;
	double t = -1.0;
	for (int i = 0; i < n; i++, x++)
		if( t < *x )
		{
			t = *x;
			k = i;
		}
	return k;

}


//-------------------------------------------------------------------
// k-center Clustering.
//-------------------------------------------------------------------
//
// Gonzalez's farthest-point clustering algorithm.
//
// OUTPUT
// ----------------
//
// MaxClusterRadius	--> maximum radius of the clusters, (rx).
// pci              --> vector of length N where the i th element is the
//                    cluster number to which the i th point belongs.
//                    pci[i] varies between 0 to K-1. 
//-------------------------------------------------------------------


void 
KCenterClustering::Cluster()
{
	
	
    int *pCenters = new int[K]; //indices of the centers.

	int    *cprev = new int[N];	    // index to the previous node
	int    *cnext = new int[N];	    // index to the next node
	int    *far2c = new int[K];     // farthest node to the center
	
	// randomly pick one node as the first center.
	srand( (unsigned)time( NULL ) );
	int nc = rand() % N;	// new center
	
	// add the ind-th node to the first center.
	pCenters[0] = nc;

	// compute the distances from each node to the first center.
	// initialize the circular linked list, the center is the
	// sentinel node.
	const double *x_nc, *x_j;
	x_nc = px + nc*d;
	x_j = px;
	for (int j = 0; j < N; x_j += d, j++)
	{
		dist_C[j] = (j==nc)? 0.0:ddist(d, x_j, x_nc);
		cnext[j] = j+1;
		cprev[j] = j-1;
	}
	cnext[N-1] = 0; // link the tail to the head.
	cprev[0] = N-1; // link the head to the tail.

	// compute the radius of the first cluster and the farthest 
	// node to the center.
	nc = idmax(N,dist_C);
	far2c[0] = nc;
	r[0] = dist_C[nc];

	for(int i = 1; i < K; i++)
	{	 
		//find the maximum of vector dist_C, i.e., find the node
		//that is farthest away from C. It is a new center.
		nc = idmax(i,r);
		nc = far2c[nc];
		pCenters[i] = nc; //add the ind-th node to the current center.
		r[i] = dist_C[nc] = 0.0;pci[nc]=i;
		far2c[i] = nc;
		cnext[cprev[nc]] = cnext[nc]; // delete nc
		cprev[cnext[nc]] = cprev[nc];
		cnext[nc] = cprev[nc] = nc; //self-loop

		//update the distances from each point to the current center.
		x_nc = px + nc*d;
		for (int j = 0; j < i; j++)
		{
			int ct_j = pCenters[j];
			x_j = px + ct_j*d;
			double dc2cq = ddist(d, x_j, x_nc) / 4;
			if (dc2cq < r[j]) // neighbor cluster
			{
				r[j] = 0.0;
				far2c[j] = ct_j;
				int k = cnext[ct_j];
				while (k != ct_j) // visit the circular linked list
				{
					int nextk = cnext[k];
					//compare the distances from new center 
					//and from current center.	
					double dist2c_k = dist_C[k];
					if ( dc2cq < dist2c_k )
					{
						
						x_j = px + k*d;
						double dd = ddist(d, x_j, x_nc);
						if ( dd < dist2c_k )
						{
							dist_C[k] = dd; // update distances to center
							pci[k]=i;
							if (r[i] < dd)  // find max r
							{
								r[i] = dd;
								far2c[i] = k;
								
							}
							cnext[cprev[k]] = nextk; // delete nextk from ct_j
							cprev[nextk] = cprev[k];
							cnext[k] = cnext[nc]; // insert nextk to nc
							cprev[cnext[nc]] = k;
							cnext[nc] = k;
							cprev[k] = nc;
							
							
						}
						else if ( r[j] < dist2c_k )
						{
							r[j] = dist2c_k;
							far2c[j] = k;
							
						}
					}
					else if ( r[j] < dist2c_k )
					{
						r[j] = dist2c_k;
						far2c[j] = k;
					} // if d < 2 r_k
					k = nextk;
				} // while k
			} // if d < 2 r
		} // for j
	} // for i

	nc = idmax(K,r);
	MaxClusterRadius=sqrt(r[nc]);

    delete []cprev;
	delete []cnext;
	delete []far2c;	
	delete []pCenters;

}

//------------------------------------------------------------------------
// Computes
// [1] the cluster centers by taking the mean of all the points
// belonging to a cluster.
// [2] the number of points in each cluster.
// [3] the radius of each cluster.
//------------------------------------------------------------------------
// NumClusters     --> number of clusters
// pClusterCenters --> pointer to the cluster centers, (d*K), 
// pNumPoints      --> pointer to the num of points in each cluster, (K). 
// pClusterRadii   --> pointer to the radius of each cluster, (K).
//------------------------------------------------------------------------

void 
KCenterClustering::ComputeClusterCenters(
	int NumClusters,
	double *pClusterCenters,
	int *pNumPoints,
	double *pClusterRadii
	)
{
	int K=NumClusters;
	
	for(int k=0; k<K; k++)
	{
		pNumPoints[k]=0;
		pClusterRadii[k]=sqrt(r[k]);
		for(int dim=0; dim<d; dim++)
		{
			pClusterCenters[(k*d)+dim]=0.0;
		}
	}

	for(int i=0; i<N; i++)
	{
		
		pNumPoints[pci[i]] += 1;

		for(int dim=0; dim<d; dim++)
		{
			pClusterCenters[(pci[i]*d)+dim] += px[(i*d)+dim];
		}
	}

	for(int k=0; k<K; k++)
	{		
		for(int dim=0; dim<d; dim++)
		{
			pClusterCenters[(k*d)+dim]=pClusterCenters[(k*d)+dim]/pNumPoints[k];
		}
	}

	

}
	
