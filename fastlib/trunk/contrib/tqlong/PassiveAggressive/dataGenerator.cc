#include <fastlib/fastlib.h>
#include "dataGenerator.h"

bool DatasetGenerator::generateNextPoint(Vector& X_out, double& y_out) {
  if (m_iCurrentPoint >= m_iLaps * m_dpData->n_points()) { // no more point
    X_out.Init(0);
    return false;
  }
  else {
    double* x = m_dpData->point(m_iCurrentPoint % m_dpData->n_points());
    X_out.Alias(x, n_features());
    y_out = x[n_features()];
    m_iCurrentPoint++;
    return true;
  }
}

void FileRowGenerator::CalculateNFeatures() {
  FILE* f = fopen(m_sFileName, "r");
  char line[10000];
  fgets(line, 10000, f);
  fclose(f);
  m_iNFeatures = 0;
  char old_char = ' ';
  for (char* p = line; *p != '\0' && *p != '\r' && *p != '\n'; p++) {
    if (old_char == ' ' || old_char == '\t' || old_char == ',')
      if (*p  != ' ' && *p != '\t' && *p != ',') m_iNFeatures++;
    old_char = *p;
  }
  m_iNFeatures--; // total numbers = n_features + 1 (the label is last)
}

bool FileRowGenerator::generateNextPoint(Vector& X_out, double& y_out) {
  double tmp;
  fscanf(m_fFile, "%lf", &tmp);                   // try a read
  if (feof(m_fFile)) {                            // if cannot read
    if (m_iCurrentLap < m_iLaps-1) {              // try next laps
      fclose(m_fFile);
      m_fFile = fopen(m_sFileName, "r");
      return generateNextPoint(X_out, y_out);
    }
    else {                                        // no laps left
      X_out.Init(0);
      return false;
    }
  }
  else {                                          // read the sample and
    X_out.Init(m_iNFeatures);                     // its label
    X_out[0] = tmp;
    for (index_t i = 1; i < m_iNFeatures; i++) {
      DEBUG_ASSERT(!feof(m_fFile));
      fscanf(m_fFile, "%lf", &tmp);
      X_out[i] = tmp;
    }
    DEBUG_ASSERT(!feof(m_fFile));
    fscanf(m_fFile, "%lf", &y_out);
    return true;
  }
}
