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
 * @file dataset_info.cc
 *
 * Implementations for DatasetInfo.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#include "../base/base.h"
#include "../col/tokenizer.h"

#include "dataset_info.h"

#include <sstream>
#include <iostream>

using namespace std;

void DatasetInfo::InitContinuous(index_t n_features, const std::string& name_in) {
  features_.reserve(n_features);

  name_ = name_in;

  for (index_t i = 0; i < n_features; i++) {
    ostringstream o;
    o << i;
    DatasetFeature f;
    f.InitContinuous(o.str());
    features_.push_back(f);
  }
}

void DatasetInfo::Init(const string& name_in) {
  name_ = name_in;
}

index_t DatasetInfo::SkipSpace_(string& s) {
  int i = 0;
  while (isspace(s[i])) {
    i++;
  }

  if (unlikely(s[i] == '%') || unlikely(s[i] == '\0')) {
    return s.length();
  }

  return i;
}

char *DatasetInfo::SkipNonspace_(char *s) {
  while (likely(*s != '\0')
      && likely(*s != '%')
      && likely(*s != ' ')
      && likely(*s != '\t')) {
    s++;
  }

  return s;
}

void DatasetInfo::SkipBlanks_(TextLineReader& reader) {
  while (reader.MoreLines() && reader.Peek()[SkipSpace_(reader.Peek())] == '\0') {
    reader.Gobble();
  }
}

success_t DatasetInfo::InitFromArff(TextLineReader& reader, const string& filename) {
  success_t result = SUCCESS_PASS;

  Init(filename);

  while (1) {
    SkipBlanks_(reader);

    string *peeked = &reader.Peek();
    vector<string> portions;

    tokenizeString(*peeked, ", \t", portions, 0, "%{", 3, true);

    if (portions.size() == 0) {
      /* empty line */
    } else if (portions[0][0] != '@') {
      reader.Error("ARFF: Unexpected @command.  Did you forget @data?");
      result = SUCCESS_FAIL;
      break;
    } else {
      if (strcasecmp(portions[0].c_str(), "@relation") == 0) {
        if (portions.size() < 2) {
          reader.Error("ARFF: @relation requires name");
          result = SUCCESS_FAIL;
        } else {
          set_name(portions[1]);
        }
      } else if (strcasecmp(portions[0].c_str(), "@attribute") == 0) {
        if (portions.size() < 3) {
          reader.Error("ARFF: @attribute requires name and type.");
          result = SUCCESS_FAIL;
        } else {
          DatasetFeature feature;
          if (portions[2][0] == '{') { //}
            feature.InitNominal(portions[1]);
            // TODO: Doesn't support values with spaces {
            tokenizeString(portions[2], ", \t", feature.value_names(), 1, "}%", 0);
            features_.push_back(feature);
          } else {
            string type(portions[2]);
            //portions[2].Trim(" \t", &type);
            if (strcasecmp(type.c_str(), "numeric") == 0
                || strcasecmp(type.c_str(), "real") == 0) {
              feature.InitContinuous(portions[1]);
              features_.push_back(feature);
            } else if (strcasecmp(type.c_str(), "integer") == 0) {
              feature.InitInteger(portions[1]);
              features_.push_back(feature);
            } else {
              reader.Error(
                  "ARFF: Only supports 'numeric', 'real', and {nominal}.");
              result = SUCCESS_FAIL;
            }
          }
        }
      } else if (strcasecmp(portions[0].c_str(), "@data") == 0) {
        /* Done! */
        reader.Gobble();
        break;
      } else {
        reader.Error("ARFF: Expected @relation, @attribute, or @data.");
        result = SUCCESS_FAIL;
        break;
      }
    }

    reader.Gobble();
  }

  return result;
}

success_t DatasetInfo::InitFromCsv(TextLineReader& reader, const std::string& filename) {
  vector<string> headers;
  bool nonnumeric = false;

  Init(filename);

  tokenizeString(reader.Peek(), ", \t", headers);

  if (headers.size() == 0) {
    reader.Error("Trying to parse empty file as CSV.");
    return SUCCESS_FAIL;
  }

  // Try to auto-detect if there is a header row
  for (index_t i = 0; i < headers.size(); i++) {
    char *end;

    (void) strtod(headers[i].c_str(), &end);

    if (end == headers[i].c_str()) {
      nonnumeric = true;
      break;
    }
  }

  if (nonnumeric) {
    for (index_t i = 0; i < headers.size(); i++) {
      DatasetFeature feature;
      feature.InitContinuous(headers[i]);
      features_.push_back(feature);
    }
    reader.Gobble();
  } else {
    for (index_t i = 0; i < headers.size(); i++) {
      DatasetFeature feature;
      ostringstream o;
      o << i;
      feature.InitContinuous(o.str());
      features_.push_back(feature);
    }
  }

  return SUCCESS_PASS;
}

success_t DatasetInfo::InitFromFile(TextLineReader& reader, const std::string& filename) {
  SkipBlanks_(reader);

  // WARNING: Safe?
  char first_char = reader.Peek()[SkipSpace_(reader.Peek())];

  if (!first_char) {
    Init();
    reader.Error("Could not parse the first line.");
    return SUCCESS_FAIL;
  } else if (first_char == '@') {
    /* Okay, it's ARFF. */
    return InitFromArff(reader, filename);
  } else {
    /* It's CSV.  We'll try to see if there are headers. */
    return InitFromCsv(reader, filename);
  }
}

success_t DatasetInfo::ReadMatrix(TextLineReader& reader, arma::mat &matrix) const {
  vector<double> linearized;
  index_t n_features = this->n_features();
  index_t n_points = 0;
  success_t retval = SUCCESS_PASS;
  bool is_done = false;

  // read through our file to find out how long it is
  index_t cur_line = reader.line_num();
  while(reader.Gobble()) { }
  matrix.set_size(n_features, reader.line_num() - cur_line + 1);

  string fname = reader.filename();
  reader.Close();
  reader.Open(fname.c_str());

  // make sure we are in the same place in our file, just in case it was passed
  // to us and we had already been some of the way through it (this is why we
  // saved the line number
  while(reader.line_num() < cur_line) {
    reader.Gobble();
  }

  while((n_points < matrix.n_cols) && !is_done && !FAILED(retval)) {
    retval = ReadPoint(reader, matrix.begin_col(n_points), is_done);
    n_points++;
  }

  if (!FAILED(retval)) {
    n_points--; // last increment was the failure, so subtract that
  }

  return retval;
}

success_t DatasetInfo::ReadPoint(TextLineReader& reader, arma::mat::col_iterator col,
    bool &is_done) const {
  // std::string is not null-terminated.
  // so half the crap in this method should fail miserably, but it seems to
  // somehow still work.  since we'll be replacing it soon, I'm going to ignore
  // it
  index_t n_features = this->n_features();
  string str;
  string::iterator pos;

  is_done = false;

  for (;;) {
    if (!reader.MoreLines()) {
      is_done = true;
      return SUCCESS_PASS;
    }

    str = reader.Peek();
    pos = str.begin();

    while (*pos == ' ' || *pos == '\t' || *pos == ',') {
      pos++;
    }

    if (unlikely(*pos == '%')) {
      reader.Gobble();
    } else {
      break;
    }
  }

  for (index_t i = 0; i < n_features; i++) {
    string::iterator next;
   
    while (pos < str.end() && (*pos == ' ' || *pos == '\t' || *pos == ',')) {
      pos++;
    }
   
    next = pos;
    while (next < str.end() && (*next != ' ' && *next != '\t' && *next != ','
        && *next != '%')) {
      next++;
    }
   
    size_t len = str.end() - pos;
    size_t cpos = pos - str.begin();
    if (!PASSED(features_[i].Parse(str.substr(cpos, len), col[i]))) {
      string::iterator end = reader.Peek().end();
      string tmp;
      tmp.assign(pos, str.end());
      reader.Error("Invalid parse: [%s]", tmp.c_str());
      return SUCCESS_FAIL;
    }

    // increment the value we are looking at
    pos = next;
  }

  reader.Gobble();

  return SUCCESS_PASS;
}

bool DatasetInfo::is_all_continuous() const {
  for (index_t i = 0; i < features_.size(); i++) {
    if (features_[i].type() != DatasetFeature::CONTINUOUS) {
      return false;
    }
  }

  return true;
}

void DatasetInfo::WriteArffHeader(TextWriter& writer) const {
  writer.Printf("@relation %s\n", name_.c_str());

  for (index_t i = 0; i < features_.size(); i++) {
    const DatasetFeature *feature = &features_[i];
    writer.Printf("@attribute %s ", feature->name().c_str());
    if (feature->type() == DatasetFeature::NOMINAL) {
      writer.Printf("{");
      for (index_t v = 0; v < feature->n_values(); v++) {
        if (v != 0) {
          writer.Write(",");
        }
        writer.Write(feature->value_name(v).c_str());
      }
      writer.Printf("}");
    } else {
      writer.Write("real");
    }
    writer.Write("\n");
  }
  writer.Printf("@data\n");
}

void DatasetInfo::WriteCsvHeader(const char *sep, TextWriter& writer) const {
  for (index_t i = 0; i < features_.size(); i++) {
    if (i != 0) {
      writer.Write(sep);
    }
    writer.Write(features_[i].name().c_str());
  }
  writer.Write("\n");
}

void DatasetInfo::WriteMatrix(const arma::mat& matrix, const char *sep,
    TextWriter& writer) const {
  for (index_t i = 0; i < matrix.n_cols; i++) {
    for (index_t f = 0; f < features_.size(); f++) {
      if (f != 0) {
        writer.Write(sep);
      }
      string str;
      features_[f].Format(matrix(f, i), str);
      writer.Write(str.c_str());
    }
    writer.Write("\n");
  }
}
