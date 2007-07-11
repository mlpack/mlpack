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
	double test_simple_lo, test_simple_hi;

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

	simple = 1;
	n = size;
	lo.Init(n,n);
	hi.Init(n,n);
	test_simple_lo = data.matrix().get(0,1);
	test_simple_hi = data.matrix().get(0,1+n);

	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) {
			double tmp_lo = data.matrix().get(i,j);
			double tmp_hi = data.matrix().get(i,j+n);

			if ( (tmp_lo != test_simple_lo || 
						tmp_hi != test_simple_hi) 
					&& i!=j) {
				simple = 0;
			}
			lo.set(i,j,tmp_lo);
			hi.set(i,j,tmp_hi);
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


success_t Matcher::Matches(const Matrix data, const Vector index, const Metric metric) const {
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
 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }
 else {
	 index_t i, j;
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
 }
 return SUCCESS_PASS; 
}


success_t Matcher::SingleMatch(const Matrix X, const Vector index, const Metric M) const {
 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }
 else {
	 index_t i,j;
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
 }
 return SUCCESS_PASS; 
}


success_t Matcher::SingleMatch(const Matrix distances) const {
	if (distances.n_rows() != n || distances.n_cols() != n ) {
		fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
		exit(1);
	}
	else {
		index_t i,j;
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
	}
	return SUCCESS_PASS;
}
/*****************************************************************************/


success_t Matcher::AnyMatch(const DataPack data, const Vector index, const Metric metric) const {
 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }
 else {
	 int ready = 0;
	 index_t i;
	 success_t match = SUCCESS_PASS;
	 Vector tau;
	 tau.Init(n);

	 if ( !PASSED(generate_first_permutation(tau)) ) {
		 fprintf(output,"Fatal error: could not generate the first permutation\n");
		 exit(1);
	 }

	 do {
		 index_t j;
		 ready = 1; // be optimistic about the current permutation

		 for (i=0;i<n;i++) {
			 for (j=i+1;j<n && PASSED(match);j++) {
				 index_t index_i = index[tau[i]], index_j = index[tau[j]];
				 double dist = metric.ComputeDistance(data,index_i,index_j);
	
				 if (dist < lo.get(i,j)) {
					 match = SUCCESS_FAIL;
				 }
				 if (dist > hi.get(i,j)) {
					 match =  SUCCESS_FAIL;
				 }
			 }
		 }

		 if ( PASSED(match) ) { // we got a match... yupii
			 return SUCCESS_PASS; 
		 }
		 else { // gotta try something new
			 success_t can_make_new_permutation = generate_next_permutation(tau);
			 ready = 0;
			 if ( !PASSED(can_make_new_permutation) ) { // we're out of permutations
				 fprintf(output,"No new permutation avaliable\n\n");
				 return SUCCESS_FAIL; // report that no match could be found
			 }
		 }
	 }
	 while (!ready);
 }
 return SUCCESS_WARN;
}


success_t Matcher::AnyMatch(const Matrix data, const Vector index, const Metric metric) const {
 if (index.length() != n ) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }
 else {
	 int ready = 0;
	 success_t match = SUCCESS_PASS;
	 Vector tau;
	 tau.Init(n);

	 if ( !PASSED(generate_first_permutation(tau)) ) {
		 fprintf(output,"Fatal error: could not generate the first permutation\n");
		 exit(1);
	 }

	 do {
		 index_t i,j;
		 ready = 1; // be optimistic about the current permutation
		 
		 for (i=0;i<n;i++) {
			 for (j=i+1;j<n && PASSED(match);j++) {
				 index_t index_i = index[tau[i]], index_j = index[tau[j]];
				 double dist = metric.ComputeDistance(data,index_i,index_j);
	
				 if (dist < lo.get(i,j)) {
					 match = SUCCESS_FAIL;
				 }
				 if (dist > hi.get(i,j)) {
					 match =  SUCCESS_FAIL;
				 }
			 }
		 }

		 if ( PASSED(match) ) { // we got a match... yupii
			 return SUCCESS_PASS;
		 }
		 else { // gotta try something new
			 success_t can_make_new_permutation = generate_next_permutation(tau);
			 ready = 0;
			 if ( !PASSED(can_make_new_permutation) ) { // out of permutations
				 return SUCCESS_FAIL; // report that no match was found
			 }
		 }
	 }
	 while (!ready);
 }
 return SUCCESS_WARN;
}


success_t Matcher::AnyMatch(const Matrix distances) const {
 if (distances.n_cols() != n || distances.n_rows() != n) {
	 fprintf(output,"Fatal error: Matcher and n-tuple dimensions don't agree!\n");
	 exit(1);
 }
 else {
	 int ready = 0;
	 success_t match = SUCCESS_PASS;
	 Vector tau;
	 Matrix dist;
	 tau.Init(n);
	 dist.Copy(distances);

	 if ( !PASSED(generate_first_permutation(tau)) ) {
		 fprintf(output,"Fatal error: could not generate the first permutation\n");
		 exit(1);
	 }

	 do {
		 index_t i,j;
		 ready = 1;
		 
		 for (i=0;i<n;i++) {
			 for (j=i+1;j<n && PASSED(match);j++) {
				 if (dist.get(i,j) < lo.get(i,j)) {
					 match = SUCCESS_FAIL;
				 }
				 if (dist.get(i,j) > hi.get(i,j)) {
					 match =  SUCCESS_FAIL;
				 }
			 }
		 }

		 if ( PASSED(match) ) {
			 return SUCCESS_PASS;
		 }
		 else {
			 ready = 0;
			 if ( !PASSED(generate_next_permutation(distances,tau,dist) ) ) {
				 return SUCCESS_FAIL;
			 }
		 }
	 }
	 while (!ready);
 }
 return SUCCESS_WARN;
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


/*****************************************************************************/
success_t generate_first_permutation(Vector &tau) {
	index_t i, n = tau.length();

	for (i=0;i<n;i++) {
		tau[i] = i;
	}

	return SUCCESS_PASS;
}


success_t generate_next_permutation(Vector &tau) {
	index_t i;
	index_t n = tau.length();
	index_t top = n-1;
	int ok_so_far = 0;

	do {
		if (ok_so_far) {
			top += 1;
		}

		tau[top] += 1;
		ok_so_far = 1;

		if (tau[top] > n-1) { 
			ok_so_far = 0;
			tau[top] = -1;
			top -= 1;
		} 
		if (top < 0) {
			return SUCCESS_FAIL;
		}
		for (i=0;i<top;i++) { 
			if (tau[top] == tau[i]) {
				ok_so_far = 0;
			}
		}
	}
	while (top < (n-1) || !ok_so_far);

	return SUCCESS_PASS;
}


success_t generate_next_permutation(Matrix dist, Vector &tau, Matrix &new_dist) {
	if ( !PASSED(generate_next_permutation(tau)) ) {
		return SUCCESS_FAIL;
	}
	else {
		index_t i,j;
		int n = tau.length();

		for (i=0;i<n;i++) {
			/* Copy column tau[i] in dist to column i in new_dist element by element */
			for (j=0;j<n;j++) {
				new_dist.set(i,j,dist.get(tau[i],j));
			}
		}
		return SUCCESS_PASS;
	}
	return SUCCESS_WARN;
}
/*****************************************************************************/

