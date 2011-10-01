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

#include "fastlib/fx/io.h"

#include "dataset.h"

#include <sstream>
#include <iostream>
#include "../fx/io.h"


size_t Dataset::n_labels() const {
  size_t i = 0;
  size_t label_row_idx = matrix_.n_rows - 1; // the last row is for labels
  size_t n_labels = 0;

  double current_label;
  
  std::vector<double> labels_list;
  labels_list.push_back(matrix_(label_row_idx,0));
  n_labels++;

  for(i = 1; i < matrix_.n_cols; i++) {
    current_label = matrix_(label_row_idx, i);
    size_t j = 0;
    for (j = 0; j < n_labels; j++) {
      if (current_label == labels_list[j]) {
        break;
      }
    }
    if(j == n_labels) { // new label
      labels_list.push_back(current_label);
      n_labels++;
    }
  }
  labels_list.clear();
  return n_labels;
}

void Dataset::GetLabels(std::vector<double> &labels_list,
                        std::vector<size_t> &labels_index,
                        std::vector<size_t> &labels_ct,
                        std::vector<size_t> &labels_startpos) const {
  size_t i = 0;
  size_t label_row_idx = matrix_.n_rows - 1; // the last row is for labels
  size_t n_points = matrix_.n_cols;
  size_t n_labels = 0;

  double current_label;

  labels_list.clear();
  labels_index.clear();
  labels_ct.clear();
  labels_startpos.clear();

  labels_index.reserve(n_points);

  std::vector<size_t> labels_temp;
  labels_temp.reserve(n_points);
  labels_temp.push_back(0);

  labels_list.push_back(matrix_(label_row_idx,0));
  labels_ct.push_back(1);
  n_labels++;

  for (i = 1; i < n_points; i++) {
    current_label = matrix_(label_row_idx, i);
    size_t j = 0;
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
  for(i = 1; i < n_labels; i++)
    labels_startpos.push_back(labels_startpos[i - 1] + labels_ct[i - 1]);

  for(i = 0; i < n_points; i++) {
    labels_index[labels_startpos[labels_temp[i]]] = i;
    labels_startpos[labels_temp[i]]++;
  }

  labels_startpos[0] = 0;
  for(i = 1; i < n_labels; i++)
    labels_startpos[i] = labels_startpos[i - 1] + labels_ct[i - 1];

  labels_temp.clear();
}


bool Dataset::InitFromFile(const char *fname) {
  TextLineReader reader;

  if ((reader.Open(fname))) {
    return InitFromFile(reader, fname);
  } else {
    matrix_ = 0.0; // 0x0 matrix
    info_.Init();
    mlpack::IO::Warn << " Could not open file " << fname << " for reading. " << std::endl;
    return false;
  }
}

bool Dataset::InitFromFile(TextLineReader& reader,
    const char *filename) {
  bool result;

  result = info_.InitFromFile(reader, filename);
  if ((result)) {
    result = info_.ReadMatrix(reader, matrix_);
  } else {
    matrix_ = 0.0; // 0x0 matrix
  }

  return result;
}


bool Dataset::WriteCsv(std::string fname, bool header) const {
  TextWriter writer;

  if (!(writer.Open(fname.c_str()))) {
    mlpack::IO::Warn << "Couldn't open " << fname.c_str() << " for writing. " << std::endl;
    return false;
  } else {
    if (header) {
      info_.WriteCsvHeader(",\t", writer);
    }
    info_.WriteMatrix(matrix_, ",\t", writer);
    return writer.Close();
  }
}

bool Dataset::WriteArff(std::string fname) const {
  TextWriter writer;

  if (!(writer.Open(fname.c_str()))) {
    mlpack::IO::Warn << "Couldn't open " << fname.c_str() << " for writing. " << std::endl;
    return false;
  } else {
    info_.WriteArffHeader(writer);
    info_.WriteMatrix(matrix_, ",", writer);
    return writer.Close();
  }
}

void Dataset::SplitTrainTest(int folds, int fold_number,
    const std::vector<size_t>& permutation,
    Dataset& train, Dataset& test) const {
  // determine number of points in test and training sets
  size_t n_test = (n_points() + folds - fold_number - 1) / folds;
  size_t n_train = n_points() - n_test;

  // initialize blank training data set
  train.InitBlank();
  train.info().InitCopy(info());

  // initialize blank testing data set
  test.InitBlank();
  test.info().InitCopy(info());

  // set sizes of training and test datasets
  train.matrix().set_size(n_features(), n_train);
  test.matrix().set_size(n_features(), n_test);

  size_t i_train = 0;
  size_t i_test = 0;
  size_t i_orig = 0;

  for (i_orig = 0; i_orig < n_points(); i_orig++) {
    double *dest;
//nk
    if ((i_orig - fold_number) % folds == 0) {
      // put this column into the test set
      dest = test.matrix().colptr(i_test);
      i_test++;
    } else {
      // put this column into the training set
      dest = train.matrix().colptr(i_train);
      i_train++;
    }

    // copy the column over in memory
    memcpy(dest,
        this->matrix().colptr(permutation[i_orig]),
        sizeof(double) * n_features());
  }

  mlpack::IO::Assert(i_train == train.n_points());
  mlpack::IO::Assert(i_test == test.n_points());
}

bool data::Load(const char *fname, arma::mat& matrix) {
  TextLineReader reader;
  DatasetInfo info; // we will ignore this, but it reads our matrix
  bool result;

  // clear our matrix
  matrix.reset();

  if ((reader.Open(fname))) {
    // read our file, since it has successfully opened
    result = info.InitFromFile(reader, fname);
    if ((result)) {
      result = info.ReadMatrix(reader, matrix);
    }
  } else {
    mlpack::IO::Warn << "Could not open file " << fname << " for reading." << std::endl;
    return false;
  }

  return result;
}

bool data::Save(const char *fname, const arma::mat& matrix) {
  TextWriter writer;

  // temporary info object that will help write our CSV
  DatasetInfo info;
  info.InitContinuous(matrix.n_rows);

  if (!(writer.Open(fname))) {
    mlpack::IO::Warn << "Couldn't open " << fname << " for writing. " << std::endl;
    return false;
  }

  info.WriteMatrix(matrix, ",\t", writer);
  return writer.Close();
}

bool data::Save(const char *fname, const arma::Col<size_t>& index_vector,
                     const arma::vec& data_vector) {
  // we need to reimplement dataset.WriteCsv with our own modifications
  // this whole thing needs to be re-done at some point to make more sense in
  // terms of the API sensibility, but this is a last-minute thing before
  // release

  // in fact, I'm not even doing this anywhere near similarly to the other way

  // ensure our vectors are the same size
  mlpack::IO::Assert(index_vector.n_elem == data_vector.n_elem);

  // open our output file
  std::ofstream out;
  out.open(fname);
  if(!out.is_open())
    return false;

  for(size_t i = 0; i < index_vector.n_elem; i++)
    out << index_vector[i] << ", " << data_vector[i] << std::endl;

  out.close();

  return true;
}
