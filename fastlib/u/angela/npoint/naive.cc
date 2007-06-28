/**
 * @author: Angela N. Grigoroaia
 * @file: naive.cc
 */

#include "fastlib/fastlib.h"
#include "globals.h"
#include "metrics.h"
#include "matcher.h"
#include "datapack.h"

#include "naive.h"


Vector naive_npoint(DataPack data, Matcher matcher, Metric metric) {
	if ( count_all_permutations && matcher.is_simple() ) {
		Vector tmp;
		tmp.Copy(symmetric_naive_npoint(data,matcher,metric));
		la::Scale(math::Factorial(matcher.size()),&tmp);
		return tmp;
	}

	if ( count_all_permutations ) {
		return asymmetric_naive_npoint(data,matcher,metric);
	}

	return symmetric_naive_npoint(data,matcher,metric);
}


Vector symmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric)
{
 index_t i, top;
 int npoints, n, nweights, dim;
 Vector index;
 Vector results;

 npoints = data.num_points();
 n = matcher.size();
 nweights = data.num_weights();
 dim = data.num_dimensions();

 index.Init(n);
 results.Init(dim + nweights);
 results.SetZero();

 /* Initializing the n-tuple index */
 for (i = 0; i < n; i++) {
	 index[i] = i;
 }
 top = n-1;
 /* Running the actual naive loop */
 do {
	 if (PASSED(matcher.Matches(data, index, metric))) {
/*
		 fprintf(output,"Found a match for indexes:");
		 for (i = 0; i < n; i++) {
			 fprintf(output," %3.0f", index[i]);
		 }
		 fprintf(output,"\n");
*/
		 results[0] += 1.0;

		 /* Updating the weighted count(s) if needed. */
		 if (nweights > 0) { 
			 Vector tmp_wcount;
			 tmp_wcount.Init(nweights);
			 tmp_wcount.SetAll(1.0);

			 for(i = 0; i < n; i++) {
				 index_t j, index_i = index[i];
				 Vector weights;
				 data.GetWeights(index_i,weights);

				 for (j = 0; j < nweights; j++) {
					 tmp_wcount[j] *= weights[j];
				 }
			 }

			 for (i = 0; i < nweights; i++) {
				 results[i+1] += tmp_wcount[i];
			 }
		 }
	 }

	 /* Generating a new valid n-tuple index */
	 index[top] = index[top] + 1;
	 while (index[top] > (npoints - n + top)) {
		 index[top] = top-1;
		 top--;
		 /* If we can't make a new index we're done. Yupiii!!! */
		 if (top < 0) {
			 return results;
		 }
		 index[top] = index[top] + 1;
	 }
	 for (i=1;i<n;i++) {
		 if (index[i] <= index[i-1]) {
			 index[i] = index[i-1] + 1;
		 }
	 }	 
	 top = n-1;
 } 
 while (1);
}


Vector asymmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric)
{
 index_t i, top;
 int npoints, n, nweights, dim;
 Vector index;
 Vector results;

 npoints = data.num_points();
 n = matcher.size();
 nweights = data.num_weights();
 dim = data.num_dimensions();

 index.Init(n);
 results.Init(dim + nweights);
 results.SetZero();

 /* Initializing the n-tuple index */
 for (i = 0; i < n; i++) {
	 index[i] = i;
 }
 top = n-1;
 /* Running the actual naive loop */
 do {
	 int is_valid = 1;
	 if (PASSED(matcher.Matches(data, index, metric))) {
/*
		 fprintf(output,"Found a match for indexes:");
		 for (i = 0; i < n; i++) {
			 fprintf(output," %3.0f", index[i]);
		 }
		 fprintf(output,"\n");
*/
		 results[0] += 1.0;

		 /* Updating the weighted count(s) if needed. */
		 if (nweights > 0) { 
			 Vector tmp_wcount;
			 tmp_wcount.Init(nweights);
			 tmp_wcount.SetAll(1.0);

			 for(i = 0; i < n; i++) {
				 index_t j, index_i = index[i];
				 Vector weights;
				 data.GetWeights(index_i,weights);

				 for (j = 0; j < nweights; j++) {
					 tmp_wcount[j] *= weights[j];
				 }
			 }

			 for (i = 0; i < nweights; i++) {
				 results[i+1] += tmp_wcount[i];
			 }
		 }
	 }

	 /* Generating a new valid n-tuple index */
	 do { // and test if the index is valid
		 if ( is_valid && top < (n-1) ) { // if we have a valid partial index
			 top += 1; // go to the next level
			 index[top] = -1; // and restart the count
		 }

		 is_valid = 1; // be optimistic about our chances
		 index[top] += 1; // increment the top value

		 if (index[top] > (npoints-1)) { // if top value is invalid
			 top--; // go back one level
			 if (top < 0) { // if previous level doesn't exist
				 return results; // we're can't make new indexes and we can return
			 }
		 }

		 for (i=1;i<top;i++) { // iterate through all previous values
			 if (index[i] == index[top]) { // if another value equals the current one
				 is_valid = 0; // the current (partial) n-tuple is not valid
			 }
		 }
	 }
	 while ( top < (n-1) && !is_valid);
 } 
 while (1);
}

