// Implementation for Data

#include "data.h"


//---------------------Data---------------------------//
///////////////
// Construction
///////////////
Data::Data(string fn, size_t port, bool random) : 
  fn_(fn), port_(port), random_(random) {
}

////////////////////////////////////////
// Determine file format and read info
////////////////////////////////////////
bool Data::ReadFileInfo() {
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
    size_t current_l, current_n_nz_ft;
    current_l = 0;
    current_n_nz_ft = 0;

    n_ex_ = 0;
    max_n_nz_ft_ = 0;
    max_l_ln_ = 0;

    rewind(fp_);
    while ((c=getc(fp_)) != EOF) {
      current_l ++;
      if (c == ':') {
	current_n_nz_ft ++;
      }
      if (c == '\n') {
	n_ex_ ++;
	if (current_l > max_l_ln_) {
	  max_l_ln_ = current_l;
	}
	if (current_n_nz_ft > max_n_nz_ft_) {
	  max_n_nz_ft_ = current_n_nz_ft;
	}
	current_l = 0;
	current_n_nz_ft = 0;
      }
    }
    //cout << "n_ex: " << n_ex_ << ", max_n_nzft: " << max_n_nz_ft_ <<
    //  ", max_l_ln: " << max_l_ln_ << endl;
  }
  rewind(fp_);

  return true;
}

void Data::ReadFromFile() {
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
  fclose(fp_);
}

void Data::InitFromSvmlight() {
  
}

void Data::InitFromCsv() {
}

void Data::InitFromArff() {
}


void Data::ReadFromPort() {
  // TODO: probabaly parallel read from different ports
}
