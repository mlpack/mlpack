// Implementation for Dataset and Data

#include "data.h"


//---------------------Dataset---------------------------//
///////////////
// Construction
///////////////
Dataset::Dataset(string fn, size_t port, bool random) : 
  fn_(fn), port_(port), random_(random) {
}

////////////////////////////////////////////////////////////////////////
// Scan through input file and count # of examples(lines), maximum #
// of features per example, and the size of the longest line.
///////////////////////////////////////////////////////////////////////
bool Dataset::ReadFileInfo() {
  ff_ = unknown;
  /*
  char *first_line = SkipSpace_(reader->Peek().begin());

  if (!first_line) {
    Init();
    reader->Error("Could not parse the first line.");
    return SUCCESS_FAIL;
  } else if (*first_line == '@') {
    // Okay, it's ARFF.
    return InitFromArff(reader, filename);
  } else {
  // It's CSV.  We'll try to see if there are headers.
    return InitFromCsv(reader, filename);
  }

  int ic;
  char c;
  size_t current_length, current_fe;
  
  current_length = 0;
  current_fe = 0;

  max_l_ln_ = 0;
  n_sp_ = 1;
  max_n_ft_ = 0;

  while ((ic=getc(fp_)) != EOF) {
    c = (char)ic;
    current_length ++;
    if (space_or_null((int)c)) {
      current_fe ++;
    }
    if (c == '\n') {
      num_examples ++;
      if (current_length > max_length_line) {
	max_length_line = current_length;
      }
      if (current_fe > max_features_example) {
	max_features_example = current_fe;
      }
      current_length = 0;
      current_fe = 0;
    }
  }
  fclose(fp);
*/
  return true;
}

void Dataset::InitFromSvmlight() {
}

void Dataset::InitFromCsv() {
}

void Dataset::InitFromArff() {
}

void Dataset::ReadFromFile() {
  // scan size of input data file
  if ((fp_ = fopen (fn_.c_str(), "r")) == NULL) {
    cout << "Cannot open input file: " << fn_ << " !"<< endl;
    exit(1);
  }
  if (!ReadFileInfo()) {
    cout << "Input file " << fn_ << "might be corrupted!" << endl;
    exit(1);
  }
  if (ff_ == svmlight) {
    InitFromSvmlight();
  }
  else if (ff_ == csv) {
    InitFromCsv();
  }
  else if (ff_ == arff) {
    InitFromArff();
  }
  else {
    cout << "Unknown input data form! Only svmlight, csv and arff are supported!";
    exit(1);
  }
}

void Dataset::ReadFromPort() {
  // TODO: probabaly parallel read from different ports
}



//---------------------Data---------------------------//
///////////////
// Construction
///////////////
Data::Data() {
}

///////////////
// Destruction
///////////////
Data::~Data() {
  if (TR_)
    delete TR_;
  if (TE_)
    delete TE_;
  if (VA_)
    delete VA_;
}
