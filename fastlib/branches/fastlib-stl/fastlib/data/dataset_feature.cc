/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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
