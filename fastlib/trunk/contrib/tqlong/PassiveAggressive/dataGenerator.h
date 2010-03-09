#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H
#include <fastlib/fastlib.h>

/**
 * DataGenerator parent class for online data generator
 * Implement n_features() and generateNextPoint() at child classes
 */
struct DataGenerator {
  index_t m_iPoints;
  index_t m_iPositives;
  index_t m_iNegatives;
public:
  DataGenerator() : m_iPoints(0), m_iPositives(0), m_iNegatives(0) {}
  virtual ~DataGenerator() {}
  /** Number of features, TO BE IMPLEMENT in child class */
  virtual int n_features() = 0;
  /** Number of points so far */
  int n_points() { return m_iPoints; }
  int n_positives() { return m_iPositives; }
  int n_negatives() { return m_iNegatives; }
  /** RETURN true if next point is available */
  bool getNextPoint(Vector& X_out, double& y_out) {
    if ( generateNextPoint(X_out, y_out) ) {
      m_iPoints++;
      if (y_out > 0) m_iPositives++;
      if (y_out < 0) m_iNegatives++;
      return true;
    }
    else return false;
  }
  /** generate next point, TO BE IMPLEMENT in child class, TRUE if next point available  */
  virtual bool generateNextPoint(Vector& X_out, double& y_out) = 0;
};


/** Online generating a Dataset */
struct DatasetGenerator : DataGenerator {
  index_t m_iLaps;
  index_t m_iCurrentPoint;
  Dataset* m_dpData;
  bool m_bFromFile;
public:
  DatasetGenerator(Dataset& data, index_t n_laps = 1) {
    m_iLaps = n_laps;
    m_iCurrentPoint = 0;
    m_dpData = &data;
    m_bFromFile = false;
  }
  DatasetGenerator(const char* filename, index_t n_laps = 1) {
    m_dpData = new Dataset;
    if ( m_dpData->InitFromFile(filename) != SUCCESS_PASS )
      DEBUG_ASSERT(0);
    m_iLaps = n_laps;
    m_iCurrentPoint = 0;
    m_bFromFile = true;
  }
  ~DatasetGenerator() {
    if (m_bFromFile) delete m_dpData;
  }

  // virtual methods
  int n_features() { return m_dpData->n_features()-1; }
  bool generateNextPoint(Vector& X_out, double& y_out);
};

/** Online generating of file rows, used in case data cannot
 *  fit into memory.
 */
struct FileRowGenerator : DataGenerator {
  const char* m_sFileName;
  index_t m_iLaps;
  FILE* m_fFile;
  index_t m_iNFeatures;
  index_t m_iCurrentLap;
public:
  FileRowGenerator(const char* filename, index_t n_laps = 1)
    : m_sFileName(filename), m_iLaps(n_laps) {
    CalculateNFeatures();
    m_fFile = fopen(m_sFileName, "r");
    DEBUG_ASSERT(m_fFile);
    m_iCurrentLap = 0;
  }
  ~FileRowGenerator() { fclose(m_fFile); }

  void CalculateNFeatures();
  
  // virtual methods
  index_t n_features() { return m_iNFeatures; }
  bool generateNextPoint(Vector& X_out, double& y_out);
};

/** Genetors for crossvalidation */
struct CrossValidationGenerator : DataGenerator {
public:
  CrossValidationGenerator(DataGenerator&, const ArrayList<index_t>&);
};

#endif /* DATA_GENERATOR_H */
