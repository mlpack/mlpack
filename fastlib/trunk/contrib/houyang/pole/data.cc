// Implementation for Data

#include "data.h"


//---------------------Data---------------------------//
///////////////
// Construction
///////////////
Data::Data(string fn, size_t port, bool random) : 
  fn_(fn), port_(port), random_(random), 
  n_ex_(0), used_ct_(0), n_ln_(0), 
  max_ft_idx_(0), max_n_nz_ft_(0), max_l_ln_(0) {
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

    n_ln_ = 0;
    max_n_nz_ft_ = 0;
    max_l_ln_ = 0;

    rewind(fp_);
    while ((c=getc(fp_)) != EOF) {
      current_l ++;
      if (c == ':') {
	current_n_nz_ft ++;
      }
      if (c == '\n') {
	n_ln_ ++;
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
    max_l_ln_ ++; // for '\n'
    //cout << "n_ln: " << n_ln_ << ", max_n_nzft: " << max_n_nz_ft_ <<
    // ", max_l_ln: " << max_l_ln_ << endl;
  }
  rewind(fp_);

  return true;
}

//////////////
// Read file
//////////////
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
  cout << "Read examples from " << fn_ << "...";
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
  // random permutation
  if (random_) {
    RandomPermute();
    cout << "random permute examples...";
  }
  cout << "done. " << endl << n_ex_ << " examples loaded." 
       << " Max dimension: " << max_ft_idx_ << "."<< endl;
  fclose(fp_);
}

/////////////////////////////
// Randomly permute examples
/////////////////////////////
void Data::RandomPermute() {
  rnd_i_.resize(n_ex_);
  for (size_t i=0; i<n_ex_; i++) {
    rnd_i_[i] = i;
  }
  for (size_t i=0; i<n_ex_; i++) {
    size_t j = rand() % n_ex_;
    swap(rnd_i_[i], rnd_i_[j]);
  }
}

///////////////////////////////
// Get an example from dataset
///////////////////////////////
Example* Data::GetExample(size_t idx) {
  if (idx >= n_ex_ || idx < 0) {
    cout << "Invalid example index: " << idx << " !" << endl;
    exit (1);
  }
  used_ct_ ++;
  if (random_) { // get an example from permuted dataset
    EXs_[rnd_i_[idx]].in_use_ = true;
    return &(EXs_[rnd_i_[idx]]);
  }
  else {
    EXs_[idx].in_use_ = true;
    return &EXs_[idx];
  }
}

/////////////////
// Print dataset
/////////////////
void Data::Print() {
  vector<Example>::iterator it;
  for (it=EXs_.begin(); it<EXs_.end(); it++) {
    it->Print();
  }
}

////////////////////////////////
// Char c is a space/null or not
////////////////////////////////
bool Data::SorN(int c) {
  if (c == 0)
    return true;
  return (isspace(c));
}

////////////////////////////////////
// Init dataset from Svmlight format
////////////////////////////////////
void Data::InitFromSvmlight() {
  char ln[max_l_ln_]; // line buffer
  char f_pair[1000], junk[1000]; // feature pair buffer
  EXs_.resize(n_ln_);
  n_ex_ = 0; max_ft_idx_ = 0;

  size_t l = 0, pos = 0;
  // ------------parse lines--------------
  while(!feof(fp_) && fgets(ln, max_l_ln_, fp_)) {
    l++;
    while(ln[pos]) {
      if(ln[pos] == '#') { // strip comment
	ln[pos] = 0;
	// TODO: save comment
      }
      if(ln[pos] == '\n') { // strip CR
	ln[pos]=0;
      }
      pos++;
    }
    // check that line starts with target value or zero, but not with feature pair
    //cout << ln << endl;
    if(sscanf(ln, "%s", f_pair) == EOF) {
      cout << "Cannot read begining of line " << l << " !"<< endl << ln << endl;
      exit(1);
    }
    pos=0;
    while((f_pair[pos] != ':') && f_pair[pos])
      pos++;
    if (f_pair[pos] == ':') {
      cout << "Line must start with label or 0 !" << endl 
	   << "Line: " << ln << endl;
      exit (1); 
    }
    // ------------get label--------------
    double lbl;
    if(sscanf(ln, "%lf", &lbl) == EOF) {
      cout << "Cannot read label at line " << l << " !"<< endl << ln << endl;
      exit(1);
    }
    EXs_[n_ex_].y_= (T_LBL)lbl;
    // -----------skip spaces and label------------
    pos=0;
    while( SorN((int)ln[pos]) )
      pos++;
    while( (!SorN((int)ln[pos])) && ln[pos] )
      pos++;
    // -----------get features------------
    long idx; double val;
    int n_r = 0; size_t n_f = 0;
    EXs_[n_ex_].Fs_.resize(max_n_nz_ft_);
    while( ((n_r=sscanf(ln+pos, "%s", f_pair)) != EOF) && 
	   (n_r > 0) && (n_f<max_n_nz_ft_) ) {
      while(SorN((int)ln[pos]))
	pos++;
      while((!SorN((int)ln[pos])) && ln[pos])
	pos++;
      if(sscanf(f_pair,"%ld:%lf%s", &idx, &val, junk)==2) {
	// it is a regular feature
	if(idx <= 0) { 
	  cout <<"Feature numbers must be larger or equal to 1!" << "Line: " << ln << endl;
	  exit (1); 
	}
	EXs_[n_ex_].Fs_[n_f].i_ = (T_IDX)idx-1; // idx starts from 0
	EXs_[n_ex_].Fs_[n_f].v_ = (T_VAL)val;
	if ((T_IDX)idx > max_ft_idx_) {
	  max_ft_idx_ = (T_IDX)idx;
	}
      }
      else {
	cout << "Cannot parse feature/value pair " 
	     << f_pair << "! Line: " << ln << endl;
	exit (1); 
      }
      n_f ++;
    }
    EXs_[n_ex_].Fs_.resize(n_f);
    n_ex_++;
  }
  EXs_.resize(n_ex_);
}

////////////////////////////////////
// Init dataset from CSV format
////////////////////////////////////
void Data::InitFromCsv() {
}

////////////////////////////////////
// Init dataset from ARFF format
////////////////////////////////////
void Data::InitFromArff() {
}


void Data::ReadFromPort() {
  // TODO: probabaly parallel read from different ports
}
