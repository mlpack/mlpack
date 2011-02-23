// Implementation for Dataset and Data

#include "data.h"


//---------------------Dataset---------------------------//
///////////////
// Construction
///////////////
Dataset::Dataset(string fn, size_t port, bool random) : 
  fn_(fn), port_(port), random_(random) {
}

////////////////////////////////////////
// Determine file format and read info
////////////////////////////////////////
bool Dataset::ReadFileInfo() {
  int c;
  // determine file format
  ff_ = unknown;
  while ((c=getc(fp_)) != EOF) {
    if (c == '@') {
      ff_ = arff;
      break;
    }
    else if (c == ':') {
      ff_ = svmlight;
      break;
    }
    else if (c == ',') {
      ff_ = csv;
      break;
    }
  }

  // Count # of examples(lines), max # of features per example,
  // and the size of the longest line
  if (ff_ == svmlight) {
    size_t current_length, current_fe;
    current_length = 0;
    current_fe = 0;

    max_l_ln_ = 0;
    n_sp_ = 1;
    max_n_ft_ = 0;

    while ((c=getc(fp_)) != EOF) {
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
    //fclose(fp);
  }

  return true;
}

void Dataset::ReadFromFile() {
  // determine file format and read other info
  if ((fp_ = fopen (fn_.c_str(), "r")) == NULL) {
    cout << "Cannot open input file: " << fn_ << " !"<< endl;
    exit(1);
  }
  if (!ReadFileInfo()) {
    cout << "Input file " << fn_ << "might be corrupted!" << endl;
    exit(1);
  }
  // read data
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

void Dataset::InitFromSvmlight() {
}

void Dataset::InitFromCsv() {
}

void Dataset::InitFromArff() {
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
