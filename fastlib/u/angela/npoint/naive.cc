/**
 * @author: Angela N. Grigoroaia
 * @file: naive.cc
 */

#include "fastlib/fastlib.h"
#include "metrics.h"
#include "matcher.h"
#include "globals.h"

double naive_npoint(DataPack data, Matcher matcher, Metric metric)
{
 double count = 0;
 index_t i, top;
 int n_points, n, n_weights, dim;
 Matrix coord, weights;
 Vector index;

 if (!PASSED(data.GetCoordinates(&coord)) || !PASSED(data.GetWeights(&weights))) {
	 /*BUBU!!!*/
 }
 index.Init(n);

 /* Initializing the n-tuple index */
 for (i = 0; i < n; i++) {
	 index[i] = i;
 }
 top = n-1;
 /* Running the actual naive loop */
 do {
	 if (PASSED(matcher.Matches(data, index, metric))) {
		 count = count + 1;
	 }

	 /* Generating a new valid n-tuple index */
	 index[top] = index[top] + 1;
	 while (index[top] > (n_points - n + top)) {
		 index[top] = top-1;
		 top--;
		 /* If we can't make a new index we're done. Yupiii!!! */
		 if (top < 0) {
			 return count;
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

