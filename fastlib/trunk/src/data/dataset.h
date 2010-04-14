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

#include "../col/col_string.h"
#include "../la/matrix.h"
#include "../math/discrete.h"
#include "../file/textfile.h"

class TextLineReader;
class TextWriter;

/**
 * Metadata about a particular dataset feature (attribute).
 *
 * Supports nominal, continuous, and integer values.
 */
class DatasetFeature {
 public:
  /**
   * Feature types supported.
   */
  enum Type {
      /** Real-valued data. */
      CONTINUOUS,
      /** Integer valued data. */
      INTEGER,
      /** Discrete data, each of which has a "name". */
      NOMINAL
  };
  
 private:
  /** Name of the feature. */
  String name_;
  /** Type of data this feature represents. */
  Type type_;
  /** If nominal, the names of each numbered value. */
  ArrayList<String> value_names_;
  
  OBJECT_TRAVERSAL(DatasetFeature) {
    OT_OBJ(name_);
    //OT_OBJ(reinterpret_cast<int &>(type_));
    OT_ENUM_EXPERT(type_, int,
      OT_ENUM_VAL(CONTINUOUS)
      OT_ENUM_VAL(INTEGER)
      OT_ENUM_VAL(NOMINAL));
    OT_OBJ(value_names_);
  }

 /**
  * Initialization common to all features.
  *
  * @param name_in the name of the feature
  */ 
 void InitGeneral(const char *name_in) {
    name_.Copy(name_in);
    value_names_.Init();
 }

 public:
  /**
   * Initialize to be a continuous feature.
   *
   * @param name_in the name of the feature
   */
  void InitContinuous(const char *name_in) {
    InitGeneral(name_in);
    type_ = CONTINUOUS;
  }

  /**
   * Initializes to an integer type.
   *
   * @param name_in the name of the feature
   */
  void InitInteger(const char *name_in) {
    InitGeneral(name_in);
    type_ = INTEGER;
  }

  /**
   * Initializes to a nominal type.
   *
   * The value_names list starts empty, so you need to add the name of
   * each feature to this.  (The dataset reading functions will do this
   * for you).
   *
   * @param name_in the name of the feature
   */
  void InitNominal(const char *name_in) {
    InitGeneral(name_in);
    type_ = NOMINAL;
  }
  
  /**
   * Creates a text version of the value based on the type.
   *
   * Continuous parameters are printed in floating point, and integers
   * are shown as integers.  For nominal, the value_name(int(value)) is
   * shown.  NaN (missing data) is always shown as '?'.
   *
   * @param value the value to format
   * @param result this will be initialized to the formatted text
   */
  void Format(double value, String *result) const;
  
  /**
   * Parses a string into the particular value.
   *
   * Integers and continuous are parsed using the normal functions.
   * For nominal, the entry 
   *
   * If an invalid parse occurs, such as a mal-formatted number or
   * a nominal value not in the list, SUCCESS_FAIL will be returned.
   *
   * @param str the string to parse
   * @param d where to store the result
   */
  success_t Parse(const char *str, double *d) const;
  
  /**
   * Gets what the feature is named.
   *
   * @return the name of the feature; for point, "Age" or "X Position"
   */
  const String& name() const {
    return name_;
  }
  
  /**
   * Identifies the type of feature.
   *
   * @return whether this is DatasetFeature::CONTINUOUS, INTEGER, or NOMINAL
   */
  Type type() const {
    return type_;
  }
  
  /**
   * Returns the name of a particular nominal value, given its index.
   *
   * The first nominal value is 0, the second is 1, etc.
   *
   * @param value the number of the value
   */
  const String& value_name(int value) const {
    DEBUG_ASSERT(type_ == NOMINAL);
    return value_names_[value];
  }
  
  /**
   * The number of nominal values.
   *
   * The values 0 to n_values() - 1 are valid.
   * This will return zero for CONTINUOUS and INTEGER types.
   *
   * @return the number of nominal values
   */
  index_t n_values() const {
    return value_names_.size();
  }
  
  /**
   * Gets the array of value names.
   *
   * Useful for creating a nominal feature yourself.
   *
   * @return a mutable array of value names
   */
  ArrayList<String>& value_names() {
    return value_names_;
  }
};

/**
 * Information describing a dataset and its features.
 */
class DatasetInfo {
 private:
  String name_;
  ArrayList<DatasetFeature> features_;

  OBJECT_TRAVERSAL(DatasetInfo) {
    OT_OBJ(name_);
    OT_OBJ(features_);
  }

 public:
  /** Gets a mutable list of all features. */
  ArrayList<DatasetFeature>& features() {
    return features_;
  }

  /** Gets information about a particular feature. */
  const DatasetFeature& feature(index_t attrib_num) const {
    return features_[attrib_num];
  }

  /** Gets the number of features. */
  index_t n_features() const {
    return features_.size();
  }
  
  /** Gets the title of the data set. */
  const char *name() const {
    return name_;
  }
  
  /** Sets the title of the data set. */
  void set_name(const char *name_in) {
    name_.Destruct();
    name_.Copy(name_in);
  }

  /**
   * Checks if all parameters are continuous.
   */
  bool is_all_continuous() const;

  /**
   * Initialize an all-continuous dataset;
   *
   * @param n_features the number of continuous features
   * @param name_in the dataset title
   */
  void InitContinuous(index_t n_features,
      const char *name_in = "dataset");

  /**
   * Initialize a custom dataset.
   *
   * This assumes you will eventually use the features() to add features.
   *
   * @param name_in the dataset title
   */
  void Init(const char *name_in = "dataset");

  /**
   * Writes the header for an ARFF file.
   */
  void WriteArffHeader(TextWriter *writer) const;
  
  /**
   * Writes header for CSV file.
   *
   * @param sep the value separator (use ",\t" for CSV)
   * @param writer the text writer to write the header line to
   */
  void WriteCsvHeader(const char *sep, TextWriter *writer) const;

  /**
   * Writes the contents of a matrix to a file.
   *
   * @param matrix the matrix
   * @param sep the separator (use ",\t" for CSV)
   * @param writer the writer to write to
   */
  void WriteMatrix(const Matrix& matrix, const char *sep,
      TextWriter *writer) const;

  /**
   * Initialize explicitly from an ARFF file.
   *
   * ARFF LIMITATIONS: Values cannot have spaces or commas, even with quotes;
   * 'string' data type not supported (nominal is supported).
   *
   * You might just use InitFromFile, which will guess the type for you.
   *
   * This will read only the header information and leave the reader at the
   * first line of data.
   */
  success_t InitFromArff(TextLineReader *reader,
      const char *filename = "dataset");
  
  /**
   * Initialize from a CSV-like file with numbers only, inferring
   * automatically that if the first row has non-numeric characters, it is
   * a header.
   *
   * InitFromFile will automatically detect this.
   */
  success_t InitFromCsv(TextLineReader *reader,
      const char *filename = "dataset");

  /**
   * Initializes the header from a file, either CSV or ARFF.
   *
   * All header lines will be gobbled, so the reader's position will be
   * left at the first line of actual data.
   * You can then read the data with matrix.
   */
  success_t InitFromFile(TextLineReader *reader,
      const char *filename = "dataset");
  /**
   * Populates a matrix from a file, given the internal data model.
   *
   * ARFF LIMITATIONS: Values cannot have spaces or commas, even with quotes;
   * 'string' data type not supported (nominal is supported).
   *
   * @param reader the reader to get lines from
   * @param matrix the matrix to store text into
   */
  success_t ReadMatrix(TextLineReader *reader, Matrix *matrix) const;

  /**
   * Reads a single vector.
   *
   * @param reader the line reader being used
   * @param point an array of length n_features()
   * @param is_done set to true if we have finished reading the file
   *        successfully -- the value of is_done is undefined if the
   *        function returns failure!
   * @return whether reading the line was successful
   */
  success_t ReadPoint(TextLineReader *reader, double *point,
      bool *is_done) const;

 private:
  char *SkipSpace_(char *s);

  char *SkipNonspace_(char *s);

  void SkipBlanks_(TextLineReader *reader);

};


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
  Matrix matrix_;
  DatasetInfo info_;
  
  OBJECT_TRAVERSAL(Dataset) {
    OT_OBJ(matrix_);
    OT_OBJ(info_);
  }
  
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
    return matrix_.n_rows();
  }
  
  /**
   * Gets the number of points/instances in the dataset.
   *
   * This corresponds to the number of <i>columns</i> in the matrix.
   *
   * @return the number of points in the dataset
   */
  index_t n_points() const {
    return matrix_.n_cols();
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
  void GetLabels(ArrayList<double> &labels_list,
                 ArrayList<index_t> &labels_index,
                 ArrayList<index_t> &labels_ct,
                 ArrayList<index_t> &labels_startpos) const;
 
  /**
   * Gets the numeric value of a particular feature and point.
   *
   * @param feature the feature index
   * @param point the point index
   */
  double get(index_t feature, index_t point) const {
    return matrix_.get(feature, point);
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
    matrix_.set(feature, point, d);
  }
  
  /**
   * Gets the "raw" form of a particular point.
   *
   * @return a C-like array of the values of a particular point
   */
  const double *point(index_t point) const {
    return matrix_.GetColumnPtr(point);
  }
  /**
   * Gets the "raw" form of a particular point.
   *
   * @return a C-like array of the values of a particular point
   */
  double *point(index_t point) {
    return matrix_.GetColumnPtr(point);
  }
  
  /**
   * Returns the matrix that stores all the data.
   */
  const Matrix& matrix() const {
    return matrix_;
  }
  /**
   * Returns the matrix that stores all the data
   * (can be used for modification).
   */
  Matrix& matrix() {
    return matrix_;
  }
  
  /**
   * Formats as text a particular location of the data set.
   *
   * @param feature the feature number
   * @param point the point index
   * @param result string that will be initialized to the formatted text
   */
  void Format(index_t feature, index_t point, String *result) const {
    info_.feature(feature).Format(get(feature, point), result);
  }
  
  /**
   * Initializer that omits the matrix and info - you must initialize
   * these yourself.
   *
   * (Although this currently does nothing, this is to future-proof your code
   * against possible changes.)
   */
  void InitBlank() {
  }
  
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
  success_t InitFromFile(TextLineReader *reader,
      const char *filename = "dataset");
  
  /**
   * Writes to a CSV file.
   *
   * @param fname name of the file
   * @param header whether to include a first line which is the titles of the
   *               data
   */
  success_t WriteCsv(const char *fname, bool header = false) const;

  /**
   * Writes to an ARFF file.
   *
   * @param fname name of the file
   */
  success_t WriteArff(const char *fname) const;

  /**
   * Initializes from a matrix copying all contents, assuming all features
   * are continuous.
   *
   * @param matrix_in data where rows are features, columns are points
   */
  void CopyMatrix(const Matrix& matrix_in) {
    InitBlank();
    matrix_.Copy(matrix_in);
    info_.InitContinuous(matrix_.n_rows());
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
  void OwnMatrix(Matrix* matrix_in) {
    InitBlank();
    matrix_.Own(matrix_in);
    info_.InitContinuous(matrix_.n_rows());
  }
  
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
  void AliasMatrix(const Matrix& matrix_in) {
    InitBlank();
    matrix_.Alias(matrix_in);
    info_.InitContinuous(matrix_.n_rows());
  }
  
  //--- Cross-validation features ---

  /**
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
   *        (use math::MakeIdentiyPermutation or math::MakeRandomPermutation)
   * @param train the training set
   * @param test the test set
   */
  void SplitTrainTest(int folds, int fold_number,
      const ArrayList<index_t>& permutation,
      Dataset *train, Dataset *test) const;
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
  success_t Load(const char *fname, Matrix *matrix);
   /**
   * Loads a matrix from a file.
   * The Matrix is Statically Initialized from a memory mapped file
   *
   * This supports only CSV Datasets
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
  success_t LargeLoad(const char *fname, GenMatrix<Precision> *matrix) {
    TextLineReader *reader = new TextLineReader();
    if (reader->Open(fname)==SUCCESS_FAIL) {
      reader->Error("Couldn't open %s", fname);
      return SUCCESS_FAIL;
    } 
    index_t dimension=0;
    String line=reader->Peek();
    ArrayList<String> result;
    result.Init();
    line.Split(",", &result);
    dimension=result.size();
    while (reader->Gobble()) {
    }
    matrix->StaticInit(dimension, reader->line_num());
    matrix->SetAll(0.0);
    delete reader;
    reader = new TextLineReader();
    reader->Open(fname);
    while (true) {
      String line=reader->Peek();
      ArrayList<String> result;
      result.Init();
      line.Split(",", &result);
      for(index_t i=0; i<result.size(); i++) {
        Precision num;
        sscanf(result[i].c_str(), "%lf", &num);
        matrix->set(i, reader->line_num()-1, (Precision)num);
      }
      if (reader->Gobble()==false) {
        break;
      }
    }
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
  success_t Save(const char *fname, const Matrix& matrix);
};

#endif
