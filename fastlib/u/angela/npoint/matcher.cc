/** 
  * @file matcher.cc	
  * @author Angela Grigoroaia
	*
  * Description Matcher code for n-point correlation computations
 **/
 
#include "matcher.h"
#include "metrics.h"
#include "globals.h"
#include "fastlib/fastlib.h"

success_t Matcher::InitFromFile(const int size, const char *file) {
	Dataset data;

	if ( !PASSED(data.InitFromFile(file)) ) {
		Init(size);
		return SUCCESS_WARN;
	}

	if (data.n_features() == 2) {
		index_t i, j;
		double lo_value = data.matrix().get(0,0), hi_value = data.matrix().get(1,0);
		Init(size);		

		for (i=0;i<n;i++) {
			for (j=i+1;j<n;j++) {
				lo.set(i,j,lo_value);
				lo.set(j,i,lo_value);
				hi.set(i,j,hi_value);
				hi.set(j,i,hi_value);
			}
		}
		return SUCCESS_PASS;
	}

	if ( data.n_points() != (2*size) || size != data.n_features() ) {
		Init(size);
		return SUCCESS_WARN;
	}

	simple = 0;
	n = size;

	return SUCCESS_PASS;
}

success_t Matcher::IsValid() const {
 index_t i, j; 
 double m_hi = 0, m_lo = 0;

 if (lo.n_rows() != n || hi.n_rows() != n) return SUCCESS_FAIL;
 
 if ( simple == 1 ) {
	 m_lo = lo.get(1,0);
	 m_hi = hi.get(1,0);
	}

 for (i=0;i<n;i++) {
	 if ( lo.get(i,i) != 0 || hi.get(i,i) != 0) return SUCCESS_FAIL;
	 for (j=1;j<n;j++) {
		 if ( lo.get(i,j) < 0 || hi.get(i,j) < 0 ) return SUCCESS_FAIL;
		 else {
			 if ( lo.get(i,j) > hi.get(i,j) ) return SUCCESS_FAIL;
			}
		 if ( simple == 1 && (lo.get(i,j) != m_lo || hi.get(i,j) != m_hi) ) return SUCCESS_FAIL;
		}
	}
 return SUCCESS_PASS;
}


success_t Matcher::Matches(const Matrix X, const Vector index, const Metric M) const {
 index_t i, j;  
 for (i=0;i<n;i++) {
	 for (j=i+1;j<n;j++) {
		 index_t index_i = index[i], index_j = index[j];
		 double dist = M.ComputeDistance(X,index_i,index_j);

		 if (dist < lo.get(i,j)) {
			 return SUCCESS_FAIL;
		 }
		 if (dist > hi.get(i,j)) {
			 return SUCCESS_FAIL;
		 }
		}
	}
 return SUCCESS_PASS; 
}

