/**
  * @file metrics.cc
  * @author Angela N. Grigoroaia
  * @date 2007.03.15
	*
  * Description: Stuff that takes care of distance computations.
**/

#include "metrics.h"
#include "globals.h"
#include "fastlib/fastlib.h"

void Metric::Init(const int size) {
	Vector v;
	v.Init(size);
	v.SetAll(1.0);

	dimension = size;
	fast_euclid = 1;
	M.Init(dimension,dimension);
	M.SetZero();
	M.SetDiagonal(v);
}


success_t Metric::InitFromFile(const int size, const char *file_name) {
	Dataset data;

	if ( !PASSED(data.InitFromFile(file_name)) ) {
	 Init(size);
	 return SUCCESS_WARN;
	}

	if ( data.n_points() <= 0 ) {
		Init(size);
		return SUCCESS_WARN;
	}
	
	if ( data.n_points() != size ) {
		Init(size);
		return SUCCESS_WARN;
	}

	if ( data.n_points() != data.n_features() ) {
 		Init(size);
		return SUCCESS_WARN;
	}
	
	if ( !is_symmetric(data.matrix()) ) {
		Init(size);
		return SUCCESS_WARN;
	}

	if ( !is_positive_definite(data.matrix()) ) {
 		Init(size);
		return SUCCESS_WARN;
	}

	dimension = size;
	M.Copy(data.matrix());
	fast_euclid = 0;

	return SUCCESS_PASS;
}


success_t is_symmetric(const Matrix M) {
	index_t i,j;

	for(i=0; i<M.n_cols(); i++) {
		for (j=0; j<i; j++) {
			if ( M.get(i,j) != M.get(j,i) ) return SUCCESS_FAIL;
		}
	}

	return SUCCESS_PASS;
}

success_t is_positive_definite(const Matrix M) {
/**
 * TODO ~> Check to see if the math addition to fastlib
 * actually implement something like this.
**/
	return SUCCESS_PASS;
}


double compute_fast_euclidean(const Vector a, const Vector b) {
	double euclid = 0;
	index_t i;

	for (i=0; i<a.length(); i++) {
		euclid += ( (a[i]-b[i]) * (a[i]-b[i]) );
	}	
	
	return sqrt(euclid);
}


double Metric::ComputeDistance(const Vector a, const Vector b) const {
	double dist = 0;
	index_t i,j;

	if (fast_euclid == 1) {
		dist = compute_fast_euclidean(a,b);
		return dist;
	}
	else {
		for (i=0; i<dimension; i++) { 
			for (j=0; j<dimension; j++) { 
				dist += (a[i]-b[i]) * (a[j]-b[j]) * M.get(i,j);
			}
		}
	}

	return sqrt(dist);
}


double Metric::ComputeDistance(const Matrix data, const index_t x, 
		const index_t	y) const {
	Vector a,b;

	data.MakeColumnVector(x,&a);
	data.MakeColumnVector(y,&b);

	return (ComputeDistance(a,b));
}


double Metric::ComputeNorm(const Vector v) const {
	Vector w;
	
	w.Init(dimension);
	w.SetZero();
	
	return (ComputeDistance(v,w)); 
}
