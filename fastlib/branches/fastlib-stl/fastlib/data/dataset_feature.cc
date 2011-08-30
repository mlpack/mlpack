/**
 * @file dataset_feature.cc
 *
 * Implementations for the DatasetFeature class.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#include "../base/base.h"

#include "dataset_feature.h"

#include <sstream>
#include <iostream>

using namespace std;

const double DatasetFeature::DBL_NAN = std::numeric_limits<double>::quiet_NaN();

void DatasetFeature::Format(double value, string& result) const {
  if (unlikely(isnan(value))) {
    result = "?";
    return;
  }
  ostringstream o;
  switch (type_) {
    case CONTINUOUS:
      if (floor(value) != value) {
        // non-integer
        o.setf(ios::scientific);
      } else {
        // value is actually an integer
        o.precision(17);
      }
      break;
    case INTEGER: 
    case NOMINAL: 
      break;
    #ifdef DEBUG
    default: abort();
    #endif
  }
  o << value;
  result = o.str();
}

success_t DatasetFeature::Parse(const std::string& str, double& d) const {
  if (unlikely(str[0] == '?') && unlikely(str[1] == '\0')) {
    d = DBL_NAN;
    return SUCCESS_PASS;
  }
  switch (type_) {
    case CONTINUOUS:
      {
        istringstream is(str);
        if((is >> d).fail())
          return SUCCESS_FAIL;
        return SUCCESS_PASS;
      }
    case INTEGER:
      {
        int i;
        istringstream is(str);
        if((is >> i).fail())
          return SUCCESS_FAIL;
        d = i;
        return SUCCESS_PASS;
      }
    case NOMINAL: {
      index_t i;
      for (i = 0; i < value_names_.size(); i++) {
        if (value_names_[i] == str) {
          d = i;
          return SUCCESS_PASS;
        }
      }
      d = DBL_NAN;
      return SUCCESS_FAIL;
    }
    default: abort();
  }
}
