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

/*****************************************************************************/

Vector naive_npoint(DataPack data, Matcher matcher, Metric metric) {
	if (count_all_permutations && matcher.is_simple()) {
		Vector tmp;
		tmp.Copy(symmetric_naive_npoint(data,matcher,metric));
		la::Scale(math::Factorial(matcher.size()),&tmp);
		return tmp;
	}

	if (count_all_permutations && !matcher.is_simple()) {
		return asymmetric_naive_npoint(data,matcher,metric);
	}

	if (!count_all_permutations) {
		return symmetric_naive_npoint(data,matcher,metric);
	}

	fprintf(output,"\nMy error: Could not select matching scheme... Aborting!\n");
	exit(1);
}

/*****************************************************************************/

Vector symmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric) {
	index_t i, top, npoints;
	int n, nweights, dim;
	Vector index;
	Vector results;

  npoints = data.num_points();
  n = matcher.size();
  nweights = data.num_weights();
  dim = data.num_dimensions();

  index.Init(n);
  results.Init(dim + nweights);
  results.SetZero();
  if ( !PASSED(generate_new_index(index,npoints)) ) {
	  fprintf(output,"Error: could not generate new index\n\n");
	  return results;
  }

  /* Running the actual naive loop */
  do {
	  if (PASSED(matcher.Matches(data, index, metric))) {
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
	  if ( !PASSED(generate_next_symmetric_index(index,npoints)) ) {
		  return results; // can't make new index => done
		}
	} 
  while (1);
}


Vector asymmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric) {
	index_t i, npoints;
	int n, nweights, dim;
  Vector index;
  Vector results;

  npoints = data.num_points();
  n = matcher.size();
  nweights = data.num_weights();
  dim = data.num_dimensions();

  index.Init(n);
  results.Init(dim + nweights);
  results.SetZero();
  index.Init(n);
  if ( !PASSED(generate_new_index(index,npoints)) ) {
	  fprintf(output,"Error: could not generate new index\n\n");
	  return results;
  }

  /* Running the actual naive loop */
  do {
	  if (PASSED(matcher.Matches(data, index, metric))) {
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
		if ( !PASSED(generate_next_asymmetric_index(index,npoints)) ) {
		  return results; // can't make new index => done
	  }
  } 
	while (1);
}

/*****************************************************************************/

success_t generate_new_index(Vector &index, index_t npoints) {
	int i, n = index.length();

	if (npoints < n) {
		fprintf(output,"Error: npoints < n\n\n");
		return SUCCESS_FAIL;
	}

	for (i = 0; i < n; i++) {
	 index[i] = i;
	}
	return SUCCESS_PASS;
}


success_t generate_next_symmetric_index(Vector &index, index_t npoints) {
	int i, ok_so_far;
	int n = index.length();
	int top = n-1;
	
	do {
		index[top] += 1;
		ok_so_far = 1;
		
		if (index[top] >= npoints) {
			index[top] = -1;
			top -= 1;
			ok_so_far = 0;

			if (top < 0) { 
				return SUCCESS_FAIL;
			}
		}
		for (i = 0; i < top && ok_so_far; i++) {
			if (index[top] <= index[i]) {
				ok_so_far = 0;
			}
		}
		if(ok_so_far) {
			top += 1;
		}
	}
	while (top < n);

	return SUCCESS_PASS;
}


success_t generate_next_asymmetric_index(Vector &index, index_t npoints) {
	int i, ok_so_far;
	int n = index.length();
	int top = n-1;
	
	do {
		index[top] += 1;
		ok_so_far = 1;
		
		if (index[top] >= npoints) {
			index[top] = -1;
			top -= 1;
			ok_so_far = 0;

			if (top < 0) { 
				return SUCCESS_FAIL;
			}
		}
		for (i = 0; i < top && ok_so_far; i++) {
			if (index[top] == index[i]) {
				ok_so_far = 0;
			}
		}
		if(ok_so_far) {
			top += 1;
		}
	}
	while (top < n);

	return SUCCESS_PASS;
}

/*****************************************************************************/

