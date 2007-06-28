/** 
  * @author Angela N. Grigoroaia
	* @file matcher.cc	
 **/

#include "fastlib/fastlib.h"
#include "globals.h"
#include "datapack.h"
#include "metrics.h"

#include "matcher.h"


/*****************************************************************************/
void Matcher::Init(const int size) {
	n = size;
	simple = 1;
	lo.Init(n,n);
	lo.SetZero();
	hi.Init(n,n);
	hi.SetZero();
}


success_t Matcher::InitFromFile(const int size, const char *file) {
	Dataset data;
	index_t i,j;

	if ( !PASSED(data.InitFromFile(file)) ) {
		Init(size);
		return SUCCESS_WARN;
	}

	if (data.n_features() == 2 && data.n_points() == 1) {
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
	lo.Init(n,n);
	hi.Init(n,n);

	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) {
			lo.set(i,j,data.matrix().get(i,j));
			hi.set(i,j,data.matrix().get(i,j+n));
		}
	}

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
/*****************************************************************************/


/*****************************************************************************/
success_t Matcher::Matches(const DataPack data, const Vector index, const Metric metric) const {
	if (simple) {
		return SingleMatch(data,index,metric);
	}
	else {
		return AnyMatch(data,index,metric);
	}
}


succes_t Matcher::Matches(const Matrix data, const Vector index, const Metric metric) const {
	if (simple) {
		return SingleMatch(data,index,metric);
	}
	else {
		return AnyMatch(data,index,metric);
	}
}


success_t Matcher::Matches(const Matrix distances) const {
	if (simple) {
		return SingleMatch(distances);
	}
	else {
		return AnyMatch(distances);
	}
}
/*****************************************************************************/


success_t Matcher::SingleMatch(const DataPack data, const Vector index, const Metric M) const {
 index_t i, j; 

 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }

 for (i=0;i<n;i++) {
	 for (j=i+1;j<n;j++) {
		 index_t index_i = index[i], index_j = index[j];
		 double dist = M.ComputeDistance(data,index_i,index_j);

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


success_t Matcher::SingleMatch(const Matrix X, const Vector index, const Metric M) const {
 index_t i, j; 

 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }

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


success_t Matcher::SingleMatch(const Matrix distances) const {
	index_t i,j;

	if (distances.n_rows() != n || distances.n_cols() != n ) {
		fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
		exit(1);
	}

	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) {
			if ( distances.get(i,j) < lo.get(i,j) ) {
				return SUCCESS_FAIL;
			}
			if ( distances.get(i,j) > hi.get(i,j) ) {
				return SUCCESS_FAIL;
			}
		}
	}
	return SUCCESS_PASS;
}
/*****************************************************************************/


success_t Matcher::AnyMatch(const DataPack data, const Vector index, const Metric metric) const {
	return SUCCESS_FAIL;
}


success_t Matcher::AnyMatch(const Matrix data, const Vector index, const Metric metric) const {
	return SUCCESS_FAIL;
}


success_t Matcher::AnyMatch(const Matrix distances) const {
	return SUCCESS_FAIL;
}
/*****************************************************************************/


/*****************************************************************************/
void Matcher::Print2File(FILE *file) const {
	index_t i,j;

	if (simple) {
		fprintf(file,"The lower bound is:  %f.\n", lo.get(1,0));
		fprintf(file,"The high bound is:  %f.\n", hi.get(1,0));
		return;
	}

	fprintf(file,"The lower bound is:\n");
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			fprintf(file," %f",lo.get(i,j)); 
		}
		fprintf(file,"\n");
	}
	fprintf(file,"The high bound is:\n");
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			fprintf(file," %f",hi.get(i,j)); 
		}
		fprintf(file,"\n");
	}
}


void Matcher::Print() const {
	index_t i,j;

	if (simple) {
		printf("The lower bound is:  %f.\n", lo.get(1,0));
		printf("The high bound is:  %f.\n", hi.get(1,0));
		return;
	}

	printf("The lower bound is:\n");
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			printf(" %f",lo.get(i,j)); 
		}
		printf("\n");
	}
	printf("The high bound is:\n");
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			printf(" %f",hi.get(i,j)); 
		}
		printf("\n");
	}
}
/*****************************************************************************/

