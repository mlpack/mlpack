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
/** @brief A class implementation for reading in dataset with special
 *         tokens: -INFINITY and INFINITY
 *
 *  @code
 *    Vector low_coord_limits, high_coord_limits;
 *    RangeReader::ReadRangeData(&low_coord_limits, &high_coord_limits,
 *                               dataset, range_data_file_name);
 *
 *    // low_coord_limits and high_coord_limits are initialized with
 *    // the low and high limits of the search window.
 *  @endcode
 */
class RangeReader {
  
 public:
  
  /** @brief Reads the orthogonal search range data from a text file
   *
   *  @param low_coord_limits An uninitialized vector which will be filled
   *                          with the lower limits of the search window.
   *  @param high_coord_limits An uninitialized vector which will be filled
   *                           with the upper limits of the search window.
   *  @param dataset The dataset used for searching. Only used for detecting
   *                 the dimensionality of the search space (i.e. how many
   *                 rows of the text file to read in).
   *  @param range_data_file_name The file to read the range data from.
   */
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
