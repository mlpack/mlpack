/** @file range_reader.h
 *
 *  This file implements a static class for reading in the search
 *  range data from a file.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#ifndef RANGE_READER_H
#define RANGE_READER_H

#include "fastlib/fastlib.h"

// text parser class for handling -INFINITY and INFINITY tokens
class RangeReader {
  
 public:
  
  static void ReadRangeData(Vector *low_coord_limits, 
			    Vector *high_coord_limits,
			    Matrix &dataset,
			    const char *range_data_file_name) {

    TextTokenizer tokenizer;
    tokenizer.Open(range_data_file_name);
    
    // Initialize the vector limits with the appropriate dimension
    low_coord_limits->Init(dataset.n_rows());
    high_coord_limits->Init(dataset.n_rows());

    for(index_t i = 0; i < low_coord_limits->length(); i++) {
      
      // read in the lower bound for this dimension
      tokenizer.Gobble();
      if((tokenizer.Current().c_str())[0] == '-' &&
	 !isdigit((tokenizer.Current().c_str())[1])) {
        tokenizer.Gobble();
        if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
	  (*low_coord_limits)[i] = -DBL_MAX;
        }
      }
      else if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
        (*low_coord_limits)[i] = DBL_MAX;
      }
      else {
        (*low_coord_limits)[i] = atof(tokenizer.Current().c_str());
      }
      
      // read in the upper bound for this dimension
      tokenizer.Gobble();
      if((tokenizer.Current().c_str())[0] == '-' &&
	 !isdigit((tokenizer.Current().c_str())[1])) {
	tokenizer.Gobble();
	if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
	  (*high_coord_limits)[i] = -DBL_MAX;
	}
      }
      else if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
	(*high_coord_limits)[i] = DBL_MAX;
      }
      else {
	(*high_coord_limits)[i] = atof(tokenizer.Current().c_str());
      }

      printf("Got [ %g %g ]\n", (*low_coord_limits)[i],
	     (*high_coord_limits)[i]);
      
      if((*low_coord_limits)[i] > (*high_coord_limits)[i]) {
	printf("Warning: you inputed the bounds [ %g %g ] for dimension %d\n",
	       (*low_coord_limits)[i], (*high_coord_limits)[i], i);
      }
    }
  }

};

#endif
