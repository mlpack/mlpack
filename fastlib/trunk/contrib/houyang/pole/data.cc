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

void Data::InitFromSvmlight() {
  char ln[max_l_ln_]; // line buffer
  char f_pair[1000]; // feature pair buffer

  EXs_.resize(n_ln_);

  size_t l = 0, x = 0, pos = 0;
  // ------------parse lines--------------
  while(!feof(fp_)) {
    l++;
    if (!fgets(ln, max_l_ln_, fp_)) {
      cout << "Cannot read line " << l << " !"<< endl << ln << endl;
      exit(1);
    }
    if (ln[0] == '\n') {
      continue;
    }
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
    EXs_[x].y_= (T_LBL)lbl;
    // -----------skip spaces------------
    pos=0;
    while( SorN((int)ln[pos]) )
      pos++;
    while( (!SorN((int)ln[pos])) && ln[pos] )
      pos++;
    // -----------get features------------
    /*
    T_IDX i; T_VAL v;
    size_t n_f = 0, n_r = 0;
    while( ((n_r=sscanf(ln+pos, "%s", f_pair)) != EOF) && 
	   (n_r > 0) && (n_f<max_n_nz_ft_) ) {
      while(SorN((int)ln[pos]))
	pos++;
      while((!space_or_null((int)line[pos])) && line[pos])
	pos++;
      if(sscanf(featurepair,"%ld:%lf%s", &w_idx, &w_val, junk)==2) {
	// it is a regular feature
	if(w_idx<=0) { 
	  cerr << "Feature numbers must be larger or equal to 1!" << endl;
	  cerr << "Line: " << line << endl;
	  exit (1); 
	}
	if((wpos>0) && ((feat[wpos-1]).widx >= w_idx)) { 
	  cerr << "Features must be in increasing order!" << endl;
	  cerr << "Line: " << line << endl;
	  exit (1); 
	}
	(feat[wpos]).widx = (T_IDX)w_idx; // feature index starts from 1
	(feat[wpos]).wval = (T_VAL)w_val; 

	if (w_idx > global.max_feature_idx)
	  global.max_feature_idx = w_idx;

	wpos++;
      }
      else {
	cout << "Cannot parse feature/value pair!!!" << endl; 
	cerr << featurepair << " in LINE: " << line << endl;
	exit (1); 
      }
    }
    */
    /*
    for () {
      EXs_[x].Fs_.PushBack(Feature( , ));
    }
    */
    x++; n_ex_++;
  }
}

void Data::InitFromCsv() {
}

void Data::InitFromArff() {
}


void Data::ReadFromPort() {
  // TODO: probabaly parallel read from different ports
}
