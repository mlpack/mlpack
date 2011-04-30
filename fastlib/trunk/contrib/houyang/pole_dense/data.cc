// Implementation for Data

#include "data.h"


//---------------------Data---------------------------//
///////////////
// Construction
///////////////
Data::Data(string fn, T_IDX port, bool random, bool center) : 
  fn_(fn), port_(port), random_(random), center_(center),
  used_ct_(0), n_ln_(0), 
  max_ft_idx_(0), max_n_nz_ft_(0), max_l_ln_(0) {
}

//////////////////////
// Number of examples
//////////////////////
T_IDX Data::Size() const {
  return EXs_.n_cols;
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
    T_IDX current_l, current_n_nz_ft;
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
  cout << "done. " << endl ;

  // center data
  if (center_) { // do data centering /sum_i x_i = 0
    cout << "Centering data to the origin... ";
    Center();
    cout << "done!"<< endl;
  }

  cout << "Number of examples loaded: "  << Size()
       << ". Max dimension: " << max_ft_idx_ << "."<< endl;
  fclose(fp_);
}

/////////////////////////////
// Randomly permute examples
/////////////////////////////
void Data::RandomPermute() {
  T_IDX n_ex = Size();
  rnd_i_.resize(n_ex);
  for (T_IDX i=0; i<n_ex; i++) {
    rnd_i_[i] = i;
  }
  for (T_IDX i=0; i<n_ex; i++) {
    T_IDX j = rand() % n_ex;
    swap(rnd_i_[i], rnd_i_[j]);
  }
}

////////////////////////////////////
// Center the data to the origin
////////////////////////////////////
void Data::Center() {
  // calc mean
  Col<T_VAL> mean = zeros< Col<T_VAL> >(max_n_nz_ft_);
  T_IDX z = Size();
  for (T_IDX i=0; i<z; i++) {
    mean = mean + EXs_.col(i);
  }
  mean = mean / z;
  // centering
  for (T_IDX i=0; i<z; i++) {
    EXs_.col(i) = EXs_.col(i) - mean;
  }
}

///////////////////////////////
// Get an example from dataset
///////////////////////////////
//void Data::GetExample(T_IDX idx, Col<T_VAL>** x_p, T_LBL* l_p) {
void Data::GetExample(T_IDX ring_idx, T_IDX &data_idx) {
  if (ring_idx >= Size() || ring_idx < 0) {
    cout << "Invalid example index: " << ring_idx << " !" << endl;
    exit (1);
  }
  used_ct_ ++;
  if (random_) { // get an example from permuted dataset
    data_idx = rnd_i_[ring_idx];
  }
  else {
    data_idx = ring_idx;
  }
}


/////////////////
// Print dataset
/////////////////
void Data::Print() const {
  cout << EXs_ << endl;
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
  EXs_ = zeros< Mat<T_VAL> >(max_n_nz_ft_, n_ln_);
  LBLs_ = zeros< Col<T_LBL> >(n_ln_);
  max_ft_idx_ = 0;
  T_IDX n_ex = 0;

  T_IDX l = 0, pos = 0;
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
    LBLs_(n_ex)= (T_LBL)lbl;
    // -----------skip spaces and label------------
    pos=0;
    while( SorN((int)ln[pos]) )
      pos++;
    while( (!SorN((int)ln[pos])) && ln[pos] )
      pos++;
    // -----------get features------------
    long idx; double val;
    int n_r = 0; T_IDX n_f = 0;
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
        EXs_(idx-1, n_ex) = (T_VAL)val; // svmlight's idx starts from 1
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
    n_ex++;
  }
  EXs_.set_size(max_n_nz_ft_, n_ex);
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

