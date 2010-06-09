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
 * @file dataset.cc
 *
 * Implementations for the dataset utilities.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#include "../base/base.h"

#include "dataset.h"

#include <sstream>
#include <iostream>


void DatasetFeature::Format(double value, std::string& result) const {
  if (unlikely(isnan(value))) {
    result = "?";
    return;
  }
  std::ostringstream o;
  switch (type_) {
    case CONTINUOUS:
      if (floor(value) != value) {
        // non-integer
        o.setf( std::ios::scientific );
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
  if( !(o << value ) )
    abort();
  result = o.str();
}

success_t DatasetFeature::Parse(const std::string& str, double *d) const {
  if (unlikely(str[0] == '?') && unlikely(str[1] == '\0')) {
    *d = DBL_NAN;
    return SUCCESS_PASS;
  }
  switch (type_) {
    case CONTINUOUS: {
//        *d = strtod(str, &end);
      std::istringstream is(str);
      if( !(is >> *d) )
        return SUCCESS_FAIL;
      return SUCCESS_PASS;
    }
    case INTEGER: {
      int i;
      std::istringstream is(str);
      if( !(is >> i) )
        return SUCCESS_FAIL;
      *d = i;
      return SUCCESS_PASS;
    }
    case NOMINAL: {
      index_t i;
      for (i = 0; i < value_names_.size(); i++) {
        if (value_names_[i] == str) {
          *d = i;
          return SUCCESS_PASS;
        }
      }
      *d = DBL_NAN;
      return SUCCESS_FAIL;
    }
    default: abort();
  }
}

// DatasetInfo ------------------------------------------------------


void DatasetInfo::InitContinuous(index_t n_features,
    const char *name_in) {
  features_.reserve(n_features);

  name_ = name_in;

  for (index_t i = 0; i < n_features; i++) {
    std::ostringstream o;
    if(!(o << i))
      abort();
    DatasetFeature f;
    f.InitContinuous(o.str());
    features_.push_back(f);
  }
}

void DatasetInfo::Init(const char *name_in) {
  name_ = name_in;
}

index_t DatasetInfo::SkipSpace_(std::string& s) {
  int i;
  while (isspace(s[i])) {
    ++i;
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

void DatasetInfo::SkipBlanks_(TextLineReader *reader) {
  while (reader->MoreLines() && reader->Peek()[SkipSpace_(reader->Peek())] == '\0') {
    reader->Gobble();
  }
}

success_t DatasetInfo::InitFromArff(TextLineReader *reader,
    const char *filename) {
  success_t result = SUCCESS_PASS;

  Init(filename);

  while (1) {
    SkipBlanks_(reader);

    std::string *peeked = &reader->Peek();
    std::vector<std::string> portions;

    tokenizeString(*peeked, "\t", portions, 0, "%", 3 ); 

    if (portions.size() == 0) {
      /* empty line */
    } else if (portions[0][0] != '@') {
      reader->Error("ARFF: Unexpected @command.  Did you forget @data?");
      result = SUCCESS_FAIL;
      break;
    } else {
//      if (portions[0].EqualsNoCase("@relation")) {
      if( !strcasecmp( portions[0].c_str(), "@relation" ) ) {
        if (portions.size() < 2) {
          reader->Error("ARFF: @relation requires name");
          result = SUCCESS_FAIL;
        } else {
          set_name(portions[1]);
        }
//      } else if (portions[0].EqualsNoCase("@attribute")) {
      } else if( !strcasecmp( portions[0].c_str(), "@attribute" ) ) {
        if (portions.size() < 3) {
          reader->Error("ARFF: @attribute requires name and type.");
          result = SUCCESS_FAIL;
        } else {
          DatasetFeature feature;
          if (portions[2][0] == '{') { //}
            feature.InitNominal(portions[1]);
            // TODO: Doesn't support values with spaces {
            tokenizeString(portions[2], ", \t", feature.value_names(), 1, "}%", 0);
            features_.push_back(feature);
          } else {
            std::string type(portions[2]);
            //portions[2].Trim(" \t", &type);
//            if (type.EqualsNoCase("numeric")
//                || type.EqualsNoCase("real")) {
            if( !strcasecmp( type.c_str(), "numeric" )
                || !strcasecmp( type.c_str(), "real" ) ) {
              feature.InitContinuous(portions[1]);
              features_.push_back(feature);
//            } else if (type.EqualsNoCase("integer")) {
          } else if( !strcasecmp( type.c_str(), "integer" ) ) {
              feature.InitContinuous(portions[1]);
              features_.push_back(feature);
            } else {
              reader->Error(
                  "ARFF: Only support 'numeric', 'real', and {nominal}.");
              result = SUCCESS_FAIL;
            }
          }
        }
//      } else if (portions[0].EqualsNoCase("@data")) {
      } else if( strcasecmp( portions[0].c_str(), "@data" ) ) {
        /* Done! */
        reader->Gobble();
        break;
      } else {
        reader->Error("ARFF: Expected @relation, @attribute, or @data.");
        result = SUCCESS_FAIL;
        break;
      }
    }

    reader->Gobble();
  }

  return result;
}

success_t DatasetInfo::InitFromCsv(TextLineReader *reader,
    const char *filename) {
  std::vector<std::string> headers;
  bool nonnumeric = false;

  Init(filename);

  tokenizeString(reader->Peek(), ", \t", headers);

  if (headers.size() == 0) {
    reader->Error("Trying to parse empty file as CSV.");
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
    reader->Gobble();
  } else {
    for (index_t i = 0; i < headers.size(); i++) {
      DatasetFeature feature;
      std::ostringstream o;
      if(!(o << i))
        abort();
      feature.InitContinuous(o.str());
      features_.push_back(feature);
    }
  }

  return SUCCESS_PASS;
}

success_t DatasetInfo::InitFromFile(TextLineReader *reader,
    const char *filename) {
  SkipBlanks_(reader);

  // WARNING: Safe?
  char *first_line = (char *)SkipSpace_(reader->Peek());

  if (!first_line) {
    Init();
    reader->Error("Could not parse the first line.");
    return SUCCESS_FAIL;
  } else if (*first_line == '@') {
    /* Okay, it's ARFF. */
    return InitFromArff(reader, filename);
  } else {
    /* It's CSV.  We'll try to see if there are headers. */
    return InitFromCsv(reader, filename);
  }
}

index_t Dataset::n_labels() const {
  index_t i = 0;
  index_t label_row_idx = matrix_.n_rows() - 1; // the last row is for labels
  index_t n_labels = 0;

  double current_label;
  
  std::vector<double> labels_list;
  labels_list.push_back(matrix_.get(label_row_idx,0));
  n_labels++;

  for (i = 1; i < matrix_.n_cols(); i++) {
    current_label = matrix_.get(label_row_idx,i);
    index_t j = 0;
    for (j = 0; j < n_labels; j++) {
      if (current_label == labels_list[j]) {
        break;
      }
    }
    if (j == n_labels) { // new label
      labels_list.push_back(current_label);
      n_labels++;
    }
  }
  labels_list.clear();
  return n_labels;
}

void Dataset::GetLabels(std::vector<double> &labels_list,
                        std::vector<index_t> &labels_index,
                        std::vector<index_t> &labels_ct,
                        std::vector<index_t> &labels_startpos) const {
  index_t i = 0;
  index_t label_row_idx = matrix_.n_rows() - 1; // the last row is for labels
  index_t n_points = matrix_.n_cols();
  index_t n_labels = 0;

  double current_label;

  // these Arraylists need initialization before-hand
  /* This faithfully replicates the effect of ArrayList.Renew().
     Is this necessary? If all we care about is initialization,
     it shouldn't be.
  */
  {
    std::vector<double> y;
    std::vector<index_t> x[3];
    labels_list.swap(y);
    labels_index.swap(x[0]);
    labels_ct.swap(x[1]);
    labels_startpos.swap(x[2]);
  }

  labels_index.reserve(n_points);

  std::vector<index_t> labels_temp;
  labels_temp.reserve(n_points);
  labels_temp[0] = 0;

  labels_list.push_back(matrix_.get(label_row_idx,0));
  labels_ct.push_back(1);
  n_labels++;

  for (i = 1; i < n_points; i++) {
    current_label = matrix_.get(label_row_idx, i);
    index_t j = 0;
    for (j = 0; j < n_labels; j++) {
      if (current_label == labels_list[j]) {
        labels_ct[j]++;
  break;
      }
    }
    labels_temp[i] = j;
    if (j == n_labels) { // new label
      labels_list.push_back(current_label); // add new label to list
      labels_ct.push_back(1);
      n_labels++;
    }
  }
  
  labels_startpos.push_back(0);
  for(i = 1; i < n_labels; i++){
    labels_startpos.push_back( labels_startpos[i-1] + labels_ct[i-1] );
  }

  for(i = 0; i < n_points; i++) {
    labels_index[labels_startpos[labels_temp[i]]] = i;
    labels_startpos[labels_temp[i]]++;
  }

  labels_startpos[0] = 0;
  for(i = 1; i < n_labels; i++)
    labels_startpos[i] = labels_startpos[i-1] + labels_ct[i-1];

  labels_temp.clear();
}

bool DatasetInfo::is_all_continuous() const {
  for (index_t i = 0; i < features_.size(); i++) {
    if (features_[i].type() != DatasetFeature::CONTINUOUS) {
      return false;
    }
  }

  return true;
}

success_t DatasetInfo::ReadMatrix(TextLineReader *reader, Matrix *matrix) const {
  std::vector<double> linearized;
  index_t n_features = this->n_features();
  index_t n_points = 0;
  success_t retval = SUCCESS_PASS;
  bool is_done;

  
  do {
    linearized.push_back(n_features);
    double *point = &linearized.back();
    retval = ReadPoint(reader, point, &is_done);
    n_points++;
  } while (!is_done && !FAILED(retval));

  if (!FAILED(retval)) {
    DEBUG_ASSERT(linearized.size() == n_features * n_points);
    DEBUG_ASSERT(linearized.size() >= n_features);
    DEBUG_ASSERT(linearized.size() % n_features == 0);
    n_points--;
    linearized.reserve(n_features * n_points);
  }

  // We can replicate this functionality, but it will not be fast.
//  linearized.Trim();

  // WHY WOULD YOU DO THIS D:<
//  matrix->Own(linearized.ReleasePtr(), n_features, n_points);

  return retval;
}

success_t DatasetInfo::ReadPoint(TextLineReader *reader, double *point,
    bool *is_done) const {
  index_t n_features = this->n_features();
  std::string str;
  std::string::iterator pos;

  *is_done = false;

  for (;;) {
    if (!reader->MoreLines()) {
      *is_done = true;
      return SUCCESS_PASS;
    }

    str = reader->Peek();
    pos = str.begin();

    while (*pos == ' ' || *pos == '\t' || *pos == ',') {
      pos++;
    }

    if (unlikely(*pos == '\0' || *pos == '%')) {
      reader->Gobble();
    } else {
      break;
    }
  }

  for (index_t i = 0; i < n_features; i++) {
    std::string::iterator next;

    while (*pos == ' ' || *pos == '\t' || *pos == ',') {
      pos++;
    }

    if (unlikely(*pos == '\0')) {
      for (std::string::iterator s = reader->Peek().begin(); s < pos; s++) { // UNDEFINED
        if (!*s) {
          *s = ',';
        }
      }
      reader->Error("I am expecting %"LI"d entries per row, "
          "but this line has only %"LI"d.",
          n_features, i);
      return SUCCESS_FAIL;
    }

    next = pos;
    while (*next != '\0' && *next != ' ' && *next != '\t' && *next != ','
        && *next != '%') {
      next++;
    }

    if (*next != '\0') {
      char c = *next;
      *next = '\0';
      if (c != '%') {
        next++;
      }
    }

    size_t len = str.end() - pos;
    size_t cpos = pos - str.begin();
    if (!PASSED(features_[i].Parse(str.substr(cpos,len), &point[i]))) {
      std::string::iterator end = reader->Peek().end();
      std::string tmp;
      tmp.assign(pos,str.end());
      for (std::string::iterator s = reader->Peek().begin();
         s < next && s < end; s++) {
        if (*s == '\0') {
          *s = ',';
        }
      }
      reader->Error("Invalid parse: [%s]", tmp.c_str());
      return SUCCESS_FAIL;
    }

    pos = next;
  }

  while (*pos == ' ' || *pos == '\t' || *pos == ',') {
    pos++;
  }

  if (*pos != '\0') {
    for (std::string::iterator s = reader->Peek().begin(); s < pos; s++) {
      if (*s == '\0') {
        *s = ',';
      }
    }
    reader->Error("Extra junk on line.");
    return SUCCESS_FAIL;
  }

  reader->Gobble();

  return SUCCESS_PASS;
}


void DatasetInfo::WriteArffHeader(TextWriter *writer) const {
  writer->Printf("@relation %s\n", name_.c_str());

  for (index_t i = 0; i < features_.size(); i++) {
    const DatasetFeature *feature = &features_[i];
    writer->Printf("@attribute %s ", feature->name().c_str());
    if (feature->type() == DatasetFeature::NOMINAL) {
      writer->Printf("{");
      for (index_t v = 0; v < feature->n_values(); v++) {
        if (v != 0) {
          writer->Write(",");
        }
        writer->Write(feature->value_name(v).c_str());
      }
      writer->Printf("}");
    } else {
      writer->Write("real");
    }
    writer->Write("\n");
  }
  writer->Printf("@data\n");
}

void DatasetInfo::WriteCsvHeader(const char *sep, TextWriter *writer) const {
  for (index_t i = 0; i < features_.size(); i++) {
    if (i != 0) {
      writer->Write(sep);
    }
    writer->Write(features_[i].name().c_str());
  }
  writer->Write("\n");
}

void DatasetInfo::WriteMatrix(const Matrix& matrix, const char *sep,
    TextWriter *writer) const {
  for (index_t i = 0; i < matrix.n_cols(); i++) {
    for (index_t f = 0; f < features_.size(); f++) {
      if (f != 0) {
        writer->Write(sep);
      }
      std::string str;
      features_[f].Format(matrix.get(f, i), str);
      writer->Write(str.c_str());
    }
    writer->Write("\n");
  }
}

// Dataset ------------------------------------------------------------------

success_t Dataset::InitFromFile(const char *fname) {
  TextLineReader reader;

  if (PASSED(reader.Open(fname))) {
    return InitFromFile(&reader, fname);
  } else {
    matrix_.Init(0, 0);
    info_.Init();
    NONFATAL("Could not open file '%s' for reading.", fname);
    return SUCCESS_FAIL;
  }
}

success_t Dataset::InitFromFile(TextLineReader *reader,
    const char *filename) {
  success_t result;

  result = info_.InitFromFile(reader, filename);
  if (PASSED(result)) {
    result = info_.ReadMatrix(reader, &matrix_);
  } else {
    matrix_.Init(0, 0);
  }

  return result;
}


success_t Dataset::WriteCsv(const char *fname, bool header) const {
  TextWriter writer;

  if (!PASSED(writer.Open(fname))) {
    NONFATAL("Couldn't open '%s' for writing.", fname);
    return SUCCESS_FAIL;
  } else {
    if (header) {
      info_.WriteCsvHeader(",\t", &writer);
    }
    info_.WriteMatrix(matrix_, ",\t", &writer);
    return writer.Close();
  }
}

success_t Dataset::WriteArff(const char *fname) const {
  TextWriter writer;

  if (!PASSED(writer.Open(fname))) {
    NONFATAL("Couldn't open '%s' for writing.", fname);
    return SUCCESS_FAIL;
  } else {
    info_.WriteArffHeader(&writer);
    info_.WriteMatrix(matrix_, ",", &writer);
    return writer.Close();
  }
}

void Dataset::SplitTrainTest(int folds, int fold_number,
    const std::vector<index_t>& permutation,
    Dataset *train, Dataset *test) const {
  index_t n_test = (n_points() + folds - fold_number - 1) / folds;
  index_t n_train = n_points() - n_test;

  train->InitBlank();
  train->info().InitCopy(info());

  test->InitBlank();
  test->info().InitCopy(info());

  train->matrix().Init(n_features(), n_train);
  test->matrix().Init(n_features(), n_test);

  index_t i_train = 0;
  index_t i_test = 0;
  index_t i_orig = 0;

  for (i_orig = 0; i_orig < n_points(); i_orig++) {
    double *dest;

    if (unlikely((i_orig - fold_number) % folds == 0)) {
      dest = test->matrix().GetColumnPtr(i_test);
      i_test++;
    } else {
      dest = train->matrix().GetColumnPtr(i_train);
      i_train++;
    }

    mem::Copy(dest,
        this->matrix().GetColumnPtr(permutation[i_orig]),
        n_features());
  }

  DEBUG_ASSERT(i_train == train->n_points());
  DEBUG_ASSERT(i_test == test->n_points());
}

success_t data::Load(const char *fname, Matrix *matrix) {
  Dataset dataset;
  success_t result = dataset.InitFromFile(fname);
  matrix->Own(&dataset.matrix());
  return result;
}

success_t data::Save(const char *fname, const Matrix& matrix) {
  Dataset dataset;
  dataset.AliasMatrix(matrix);
  return dataset.WriteCsv(fname);
}

