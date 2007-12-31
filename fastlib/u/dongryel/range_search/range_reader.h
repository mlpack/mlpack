#ifndef RANGE_READER_H
#define RANGE_READER_H

#include "fastlib/fastlib_int.h"

// text parser class for handling -INFINITY and INFINITY tokens
class RangeReader {
  
 public:
  
  static void ReadRangeData(ArrayList<DRange> &range) {

    TextTokenizer tokenizer;
    const char *dname = fx_param_str(NULL, "range", "range.ds");
    const double factor = fx_param_double(NULL, "factor", 1);
    const double offset = fx_param_double(NULL, "offset", 0);
    tokenizer.Open(dname);
    
    for(index_t i = 0; i < range.size(); i++) {
      
      // read in the lower bound for this dimension
      tokenizer.Gobble();
      if((tokenizer.Current().c_str())[0] == '-' &&
	 !isdigit((tokenizer.Current().c_str())[1])) {
        tokenizer.Gobble();
        if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
          range[i].lo = -MAXDOUBLE;
        }
      }
      else if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
        range[i].lo = MAXDOUBLE;
      }
      else {
        range[i].lo = atof(tokenizer.Current().c_str());
      }
      
      // read in the upper bound for this dimension
      tokenizer.Gobble();
      if((tokenizer.Current().c_str())[0] == '-' &&
	 !isdigit((tokenizer.Current().c_str())[1])) {
	tokenizer.Gobble();
	if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
	  range[i].hi = -MAXDOUBLE;
	}
      }
      else if(strncmp(tokenizer.Current().c_str(), "INFINITY", 8) == 0) {
	range[i].hi = MAXDOUBLE;
      }
      else {
	range[i].hi = atof(tokenizer.Current().c_str());
      }
      
      // apply offset
      range[i].lo -= offset;
      range[i].hi -= offset;
      range[i].hi = range[i].lo + (range[i].hi - range[i].lo) * factor;

      printf("Got [ %g %g ]\n", range[i].lo, range[i].hi);
      
      if(range[i].lo > range[i].hi) {
	printf("Warning: you inputed the bounds [ %g %g ] for dimension %d\n",
	       range[i].lo, range[i].hi, i);
      }
    }
  }

};

#endif
