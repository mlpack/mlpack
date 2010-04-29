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

#include "dataset_info.h"

void DatasetInfo::InitContinuous(index_t n_features, const char *name_in) {
  features_.Init(n_features);

  name_.Copy(name_in);

  for (index_t i = 0; i < n_features; i++) {
    String feature_name;
    feature_name.InitSprintf("feature_%d", int(i));
    features_[i].InitContinuous(feature_name);
  }
}

void DatasetInfo::Init(const char *name_in) {
  features_.Init();
  name_.Copy(name_in);
}

char *DatasetInfo::SkipSpace_(char *s) {
  while (isspace(*s)) {
    s++;
  }

  if (unlikely(*s == '%') || unlikely(*s == '\0')) {
    return s + strlen(s);
  }

  return s;
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
  while (reader.MoreLines() && *SkipSpace_(reader.Peek().begin()) == '\0') {
    reader.Gobble();
  }
}

success_t DatasetInfo::InitFromArff(TextLineReader& reader, const char *filename) {
  success_t result = SUCCESS_PASS;

  Init(filename);

  while (1) {
    SkipBlanks_(reader);

    String *peeked = &reader.Peek();
    ArrayList<String> portions;

    portions.Init();
    peeked->Split(0, " \t", "%", 3, &portions);

    if (portions.size() == 0) {
      /* empty line */
    } else if (portions[0][0] != '@') {
      reader.Error("ARFF: Unexpected @command.  Did you forget @data?");
      result = SUCCESS_FAIL;
      break;
    } else {
      if (portions[0].EqualsNoCase("@relation")) {
        if (portions.size() < 2) {
          reader.Error("ARFF: @relation requires name");
          result = SUCCESS_FAIL;
        } else {
          set_name(portions[1]);
        }
      } else if (portions[0].EqualsNoCase("@attribute")) {
        if (portions.size() < 3) {
          reader.Error("ARFF: @attribute requires name and type.");
          result = SUCCESS_FAIL;
        } else {
          if (portions[2][0] == '{') { //}
            DatasetFeature *feature = &features_.PushBack();

            feature->InitNominal(portions[1]);
            // TODO: Doesn't support values with spaces {
            portions[2].Split(1, ", \t", "}%", 0, &feature->value_names());
          } else {
            String type(portions[2]);
            //portions[2].Trim(" \t", &type);
            if (type.EqualsNoCase("numeric")
                || type.EqualsNoCase("real")) {
              features_.PushBack().InitContinuous(portions[1]);
            } else if (type.EqualsNoCase("integer")) {
              features_.PushBack().InitInteger(portions[1]);
            } else {
              reader.Error(
                  "ARFF: Only supports 'numeric', 'real', and {nominal}.");
              result = SUCCESS_FAIL;
            }
          }
        }
      } else if (portions[0].EqualsNoCase("@data")) {
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

success_t DatasetInfo::InitFromCsv(TextLineReader& reader, const char *filename) {
  ArrayList<String> headers;
  bool nonnumeric = false;

  Init(filename);

  headers.Init();
  reader.Peek().Split(", \t", &headers);

  if (headers.size() == 0) {
    reader.Error("Trying to parse empty file as CSV.");
    return SUCCESS_FAIL;
  }

  // Try to auto-detect if there is a header row
  for (index_t i = 0; i < headers.size(); i++) {
    char *end;

    (void) strtod(headers[i], &end);

    if (end != headers[i].end()) {
      nonnumeric = true;
      break;
    }
  }

  if (nonnumeric) {
    for (index_t i = 0; i < headers.size(); i++) {
      features_.PushBack().InitContinuous(headers[i]);
    }
    reader.Gobble();
  } else {
    for (index_t i = 0; i < headers.size(); i++) {
      String name;
      name.InitSprintf("feature%"LI"d", i);
      features_.PushBack().InitContinuous(name);
    }
  }

  return SUCCESS_PASS;
}

success_t DatasetInfo::InitFromFile(TextLineReader& reader, const char *filename) {
  SkipBlanks_(reader);

  char *first_line = SkipSpace_(reader.Peek().begin());

  if (!first_line) {
    Init();
    reader.Error("Could not parse the first line.");
    return SUCCESS_FAIL;
  } else if (*first_line == '@') {
    /* Okay, it's ARFF. */
    return InitFromArff(reader, filename);
  } else {
    /* It's CSV.  We'll try to see if there are headers. */
    return InitFromCsv(reader, filename);
  }
}

success_t DatasetInfo::ReadMatrix(TextLineReader& reader, arma::mat &matrix) const {
  ArrayList<double> linearized;
  index_t n_features = this->n_features();
  index_t n_points = 0;
  success_t retval = SUCCESS_PASS;
  bool is_done;

  // read through our file to find out how long it is
  index_t cur_line = reader.line_num();
  while(reader.Gobble()) { }
  matrix.set_size(n_features, reader.line_num() - cur_line + 1);

  char *fname = strncpy(new char[strlen(reader.filename()) + 1], reader.filename(),
    strlen(reader.filename()) + 1);
  reader.Close();
  reader.Open(fname);
  delete[] fname;

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
    DEBUG_ASSERT(n_points == matrix.n_rows);
  }

  return retval;
}

success_t DatasetInfo::ReadPoint(TextLineReader& reader, arma::mat::col_iterator col,
    bool &is_done) const {
  index_t n_features = this->n_features();
  char *pos;

  is_done = false;

  for (;;) {
    if (!reader.MoreLines()) {
      is_done = true;
      return SUCCESS_PASS;
    }

    pos = reader.Peek().begin();

    while (*pos == ' ' || *pos == '\t' || *pos == ',') {
      pos++;
    }

    if (unlikely(*pos == '\0' || *pos == '%')) {
      reader.Gobble();
    } else {
      break;
    }
  }

  for (index_t i = 0; i < n_features; i++) {
    char *next;
   
    while (*pos == ' ' || *pos == '\t' || *pos == ',') {
      pos++;
    }
   
    if (unlikely(*pos == '\0')) {
      for (char *s = reader.Peek().begin(); s < pos; s++) {
        if (!*s) {
          *s = ',';
        }
      }
      reader.Error("I am expecting %"LI"d entries per row, "
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
   
    if (!PASSED(features_[i].Parse(pos, *col))) {
      char *end = reader.Peek().end();
      String tmp;
      tmp.Copy(pos);
      for (char *s = reader.Peek().begin(); s < next && s < end; s++) {                                                                                                                          
        if (*s == '\0') {
          *s = ',';
        }
      }
      reader.Error("Invalid parse: [%s]", tmp.c_str());
      return SUCCESS_FAIL;
    }

    // increment the value we are looking at
    col++;

    pos = next;
  }

  while (*pos == ' ' || *pos == '\t' || *pos == ',') {
    pos++;
  }

  if (*pos != '\0') {
    for (char *s = reader.Peek().begin(); s < pos; s++) {
      if (*s == '\0') {
        *s = ',';
      }
    }
    reader.Error("Extra junk on line.");
    return SUCCESS_FAIL;
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
      String str;
      features_[f].Format(matrix(f, i), &str);
      writer.Write(str);
    }
    writer.Write("\n");
  }
}
