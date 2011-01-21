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
 * @file dataset.h
 *
 * Generic datasets and associated utilities.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#ifndef DATA_DATASET_H
#define DATA_DATASET_H

#include <armadillo>
#include <string>
#include <vector>

#include "../la/matrix.h"
#include "../math/discrete.h"
#include "../file/textfile.h"
#include "../col/tokenizer.h"

#include "dataset_feature.h"
#include "dataset_info.h"

class TextLineReader;
class TextWriter;

/**
 * Most generic dataset type.
 *
 * To allow polymorphic usage, everything is internally stored as doubles.
 * If your data are discrete, the integer values are stored in the doubles.
 *
 * (Implementation note: To allow efficient polymorphic use of mixed data,
 * we found it is far more efficient to store these as doubles than to
 * introduce any sort of casting or pointer arithmetic overhead.  Space
 * may be an issue, especially if this is Boolean data.)
 */
class Dataset {
 private:
  arma::mat matrix_;
  DatasetInfo info_;
  
 public:
  /**
   * Metadata about the feature types and names for the dataset.
   *
   * Each dataset has a name, and each feature is described by a type
   * (continuous, etc) and a name, modelled loosely over the ARFF format.
   * See the DatasetInfo documentation for more information.
   * 
   * @return metadata about the dataset
   */
  const DatasetInfo& info() const {
    return info_;
  }
  
  /**
   * Metadata about the feature types and names for the dataset.
   * @return a mutable reference to the dataset
   */
  DatasetInfo& info() {
    return info_;
  }
  
  /**
   * Gets the number of features/attributes the dataset has.
   *
   * This corresponds to the number of <i>rows</i> in the matrix.
   *
   * @return the number of features, or variables, in the dataset
   */
  index_t n_features() const {
    return matrix_.n_rows;
  }

  /**
   * Gets the number of points/instances in the dataset.
   *
   * This corresponds to the number of <i>columns</i> in the matrix.
   *
   * @return the number of points in the dataset
   */
  index_t n_points() const {
    return matrix_.n_cols;
  }

  /**
   * Gets the number of labels in a labeled dataset.
   *
   * This corresponds to the number of different items of the
   * n_features-th row (last row) in the matrix.
   *
   * @return the number of labels in the dataset
   */
  index_t n_labels() const;

  /**
   * Gets a list and indicies of labels in a labeled dataset.
   *
   * The list corresponds to the different items of the n_features-th
   * row (last row) in the matrix. The indices is arranged after
   * grouping labels of the same class, i.e. (class_1
   * class_2...class_k), each item indicate the position of the label
   * in the dataset.
   *
   * @param labels_list a list of labels in the dataset. e.g. [0.0,1.0,2.0]
   *        for a 3-class dataset
   * @param labels_index the label indices of each data point. e.g.
   *        [(c1)[0,5,6,7,10,13,17],
   *         (c2)[1,2,4,8,9],
   *         (c3)[3,11,12,14,15,16,18,19]]
   * @param labels_ct numbers of point in each label class. e.g.
   *        [7,5,8]
   * @param labels_startpos start positions of each label class in
   *        labels_index. e.g. [0,7,12]                                                                                                                            
   */
  void GetLabels(std::vector<double> &labels_list,
      std::vector<index_t> &labels_index,
      std::vector<index_t> &labels_ct,
      std::vector<index_t> &labels_startpos) const;

  /**
   * Gets the numeric value of a particular feature and point.
   *
   * @param feature the feature index
   * @param point the point index
   */
  double get(index_t feature, index_t point) const {
    return matrix_(feature, point);
  }

  /**
   * Gets the integer value of a particular feature and point.
   */
  int get_int(index_t feature, index_t point) const {
    double d = get(feature, point);
    int i = int(d);
    DEBUG_ASSERT(d == double(i));
    return i;
  }

  /**
   * Modifies a value in the dataset.
   *
   * @param feature the feature number
   * @param point the point index
   * @param d the numeric value to set it to
   */
  void set(index_t feature, index_t point, double d) {
    matrix_(feature, point) = d;
  }

  /**
   * Gets the "raw" form of a particular point.
   *
   * @return a C-like array of the values of a particular point
   */
  const arma::rowvec point(index_t point) const {
    return matrix_.row(point);
  }
  /**
   * Gets the "raw" form of a particular point.
   *
   * @return a C-like array of the values of a particular point
   */
  arma::rowvec point(index_t point) {
    return matrix_.row(point);
  }

  /**
   * Returns the matrix that stores all the data.
   */
  const arma::mat& matrix() const {
    return matrix_;
  }
  /**
   * Returns the matrix that stores all the data
   * (can be used for modification).
   */
  arma::mat& matrix() {
    return matrix_;
  }

  /**
   * Formats as text a particular location of the data set.
   *
   * @param feature the feature number
   * @param point the point index
   * @param result string that will be initialized to the formatted text
   */
  void Format(index_t feature, index_t point, std::string& result) const {
    info_.feature(feature).Format(matrix_(feature, point), result);
  }

  /**
   * Initializer that omits the matrix and info - you must initialize
   * these yourself.
   *
   * (Although this currently does nothing, this is to future-proof your code
   * against possible changes.)
   */
  void InitBlank() { }

  /**
   * Reads in an ARFF or CSV/WSV file.
   *
   * ARFF LIMITATIONS: Values cannot have spaces or commas, even with quotes;
   * 'string' data type not supported (nominal is supported).
   *
   * @param fname the name of an ARFF, CSV, or whitespace-separated
   */
  success_t InitFromFile(const char *fname);

  /**
   * Reads in an ARFF or CSV/WSV file.
   *
   * ARFF LIMITATIONS: Values cannot have spaces or commas, even with quotes;
   * 'string' data type not supported (nominal is supported).
   *
   * @param reader a line reader opened on a CSV or WSV or ARFF file
   * @param filename a title given to this data set, doesn't necessarily
   *        need to be anything significant
   */
  success_t InitFromFile(TextLineReader& reader,
      const char *filename = "dataset");

  /**
   * Writes to a CSV file.
   *
   * @param fname name of the file
   * @param header whether to include a first line which is the titles of the
   *               data
   */
  success_t WriteCsv(std::string fname, bool header = false) const;

  /**
   * Writes to an ARFF file.
   *
   * @param fname name of the file
   */
  success_t WriteArff(std::string fname) const;

  /**
   * Initializes from a matrix copying all contents, assuming all features
   * are continuous.
   *
   * @param matrix_in data where rows are features, columns are points
   */
  void CopyMatrix(const arma::mat& matrix_in) {
    InitBlank();
    matrix_ = matrix_in;
    info_.InitContinuous(matrix_.n_rows);
  }

  /**
   * Initializes by becoming the owner of an existing matrix, assuming
   * all features are continuous.
   *
   * By becoming the owner of the matrix, it means that the matrix will be
   * freed when the Dataset falls out of scope.  See Matrix class for
   * details about the Own function.
   *
   * @param matrix_in data where rows are features, columns are points
   */
//    void OwnMatrix(Matrix* matrix_in) {
//      InitBlank();
//      matrix_.Own(matrix_in);
//      info_.InitContinuous(matrix_.n_rows());
//    }

  /**
   * Initializes as an alias or mirror of an existing matrix, assuming
   * all features are continuous.
   *
   * This does not copy the matrix but instead refers to an existing
   * matrix, and changes in one will be reflected in the other.  Make
   * sure the other matrix does not fall out of scope and get freed!
   *
   * @param matrix_in data where rows are features, columns are points
   */
//    void AliasMatrix(const arma::mat& matrix_in) {
//      InitBlank();
      // use the memory pointer of the other matrix directly; do not free it when
      // we are done
//      matrix_(matrix_in.memptr(), matrix_in.n_rows, matrix_in.n_cols, false);
//      info_.InitContinuous(matrix_.n_rows);
//    }

  //--- Cross-validation features ---

  /*
   * Creates a training and test dataset for k-fold cross validation.
   *
   * The test set will be approximately n_points() / folds, and the
   * training set will be all remaining points.  This takes as an argument
   * a permutation to allow use of consistent random permutations.  If
   * an identity permutation is used, the split will be performed strided.
   *
   * @param folds the number of folds being used
   * @param fold_number the fold number, 0 to folds - 1
   * @param permutation the permutation to use, the same size as n_points()
   *        (use math::MakeIdentityPermutation or math::MakeRandomPermutation)
   * @param train the training set
   * @param test the test set
   */
  void SplitTrainTest(int folds, int fold_number,
      const std::vector<index_t>& permutation,
      Dataset& train, Dataset& test) const;
};

/**
 * Miscellaneous dataset-related routines.
 */
namespace data {
  /**
   * Loads a matrix from a file.
   *
   * This supports any type the Dataset class supports with the
   * InitFromFile function: CSV and ARFF.
   *
   * @code
   * Matrix A;
   * data::Load("foo.csv", &A);
   * @endcode
   *
   * @param fname the file name to load
   * @param matrix a pointer to an uninitialized matrix to load
   */
  success_t Load(const char *fname, arma::mat& matrix);

  /**
   * Loads a matrix from a file.
   * The matrix is statically initialized from a memory mapped file.
   *
   * This supports only CSV datasets.
   *
   * @code
   * Matrix A;
   * data::LargeLoad("foo.csv", &A);
   * @endcode
   *
   * @param fname the file name to load
   * @param matrix a pointer to an uninitialized matrix to load
   */
  template<typename Precision>
  success_t LargeLoad(const char *fname, arma::Mat<Precision>& matrix) {
    // open our file
    TextLineReader *reader = new TextLineReader();
    if (reader->Open(fname) == SUCCESS_FAIL) {
      reader->Error("Couldn't open %s", fname);
      return SUCCESS_FAIL;
    }

    // find dimensionality
    index_t dimension = 0;
    std::string line = reader->Peek();
    std::vector<std::string> result;
    tokenizeString(line, ",", result);

    dimension = result.size();

    // count lines in file
    while(reader->Gobble()) { }

    // resize matrix to correct size and set all to 0
    matrix.zeros(dimension, reader->line_num());

    delete reader;

    // second pass through file: fill matrix
    reader = new TextLineReader();
    reader->Open(fname);

    do {
      // parse this line
      std::string line = reader->Peek();
      std::vector<std::string> result;

      tokenizeString(line, ",", result);
      for(index_t i = 0; i < result.size(); i++) {
        Precision num;
        sscanf(result[i].c_str(), "%lf", &num);
        matrix(i, reader->line_num() - 1) = (Precision) num;
      }
    } while(reader->Gobble()); // break when we can't read more
    
    return SUCCESS_PASS;  
  }

  /**
   * Saves a matrix to a file.
   *
   * This saves in CSV format that MATLAB and Excel can handle.
   *
   * @code
   * Matrix matrix_to_save;
   * ... matrix_to_save contains the values you want to save
   * data::Save("mymatrix.csv", matrix_to_save);
   * @endcode
   *
   * @param fname the file name to load
   * @param matrix a pointer to an uninitialized matrix to load
   */
  success_t Save(const char *fname, const arma::mat& matrix);
  success_t Save(const char *fname, const arma::Col<index_t>& index_vector,
      const arma::vec& data_vector);
};

#endif
