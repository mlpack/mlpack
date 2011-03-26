// Implementation for Sparse Linear Algebra

#include "sparsela.h"

//-------------------------Feature----------------------------//

//////////////////////////
// Create an empty feature
//////////////////////////
Feature::Feature() {
}

//////////////////////////
// Create a feature
//////////////////////////
Feature::Feature(const T_IDX i, const T_VAL v) : i_(i), v_(v) {
}

//////////////////////////
// Deconstruct feature
//////////////////////////
Feature::~Feature() {
}

//---------------------General Sparse Vector-----------------//

////////////////////////////////
// Create an empty sparse vector
////////////////////////////////
Svector::Svector() {
}

//////////////////////////////////////////////
// Create a sparse vector from given features
//////////////////////////////////////////////
Svector::Svector(const vector<Feature> Fs) : Fs_(Fs) {
}

////////////////////////////////////////////
// Create a (dense) vector of constant value
////////////////////////////////////////////
Svector::Svector(const T_IDX n_f, const T_VAL v) : Fs_(n_f, Feature()){
  T_IDX sz = Fs_.size();
  for (T_IDX i=0; i<sz; i++) {
    Fs_[i] = Feature(i, v);
  }
}

//////////////////////////
// Deconstruct Svector
//////////////////////////
Svector::~Svector() {
}

//////////////////////
// Number of faetures 
//////////////////////
size_t Svector::Size() {
  return Fs_.size();
}

/////////////////////////////////////////////
// Set to be a vector of given constant value
/////////////////////////////////////////////
void Svector::SetAll(const T_VAL v) {
  T_IDX sz = Fs_.size();
  for (T_IDX i=0; i<sz; i++) {
    Fs_[i] = Feature(i, v); // idx starts from 1
  }
}

//////////////////////////////////////////////////////////////
// Set to be a vector of given lenght and given constant value
//////////////////////////////////////////////////////////////
void Svector::SetAllResize(const T_IDX n_f, const T_VAL v) {
  Fs_.resize(n_f);
  for (T_IDX i=0; i<n_f; i++) {
    Fs_[i] = Feature(i, v); // idx starts from 1
  }
}

///////////////////////////////
// Push one feature at the end
///////////////////////////////
void Svector::PushBack(const Feature& F) {
  Fs_.push_back(F);
}

///////////////////////////////////////////
// Insert one feature at a given position p
///////////////////////////////////////////
void Svector::InsertOne(const T_IDX p, const Feature& F) {
  Fs_.insert(Fs_.begin() + p, F);
}

///////////////////////////////////////////
// Erase one feature at a given position p
///////////////////////////////////////////
void Svector::EraseOne(const T_IDX p) {
  Fs_.erase(Fs_.begin() + p);
}

////////////////
// Clear content
////////////////
void Svector::Clear() {
  Fs_.clear();
}

/////////////////
// Print content
/////////////////
void Svector::Print() {
  if (Fs_.empty()) {
    cout << "null" << endl;
  }
  else {
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      cout << it->i_ << ":" << it->v_ << " ";
    }
    cout << endl;
  }
}

//////////////////////////////////
// Copy from a given sparse vector
//////////////////////////////////
Svector& Svector::operator=(const Svector& v) {
  Fs_ = v.Fs_;
  return *this;
}

///////////////////////////////
// Sparse vector add: w += x
///////////////////////////////
Svector& Svector::operator+=(const Svector& x) {
  if (x.Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return *this;
  }
  else if (Fs_.empty()) { // w: all-0
    Fs_ = x.Fs_;
    return *this;
  }
  else { // neither w nor x is of all-0
    size_t ct_w = 0, ct_x = 0;
    while (ct_w<Fs_.size() || ct_x<x.Fs_.size()) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
	Fs_.push_back(x.Fs_[ct_x]);
	++ct_w; ++ct_x;
      }
      else if (ct_x == x.Fs_.size()) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x.Fs_[ct_x].i_) {
	  Fs_[ct_w].v_ += x.Fs_[ct_x].v_;
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x.Fs_[ct_x].i_) {
	  Fs_.insert(Fs_.begin()+ct_w, x.Fs_[ct_x]);
	  ++ct_w; ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
	  ++ct_w;
	}
      }
    }
    return *this;
  }
}

////////////////////////////////////////
// Sparse vector subtraction: w -= x
////////////////////////////////////////
Svector& Svector::operator-=(const Svector& x) {
  if (x.Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return *this;
  }
  else if (Fs_.empty()) { // w: all-0
    Fs_ = x.Fs_;
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      it->v_ = - it->v_;
    }
    return *this;
  }
  else { // neither w nor x is of all-0
    size_t ct_w = 0, ct_x = 0;
    while (ct_w<Fs_.size() || ct_x<x.Fs_.size()) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
	Fs_.push_back(Feature(x.Fs_[ct_x].i_, -x.Fs_[ct_x].v_));
	++ct_w; ++ct_x;
      }
      else if (ct_x == x.Fs_.size()) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x.Fs_[ct_x].i_) {
	  Fs_[ct_w].v_ -= x.Fs_[ct_x].v_;
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x.Fs_[ct_x].i_) {
	  Fs_.insert(Fs_.begin()+ct_w, Feature(x.Fs_[ct_x].i_, -x.Fs_[ct_x].v_));
	  ++ct_w; ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
	  ++ct_w;
	}
      }
    }
    return *this;
  }
}

/////////////////////////////////////
// Sparse vector scaling: w *= a
/////////////////////////////////////
Svector& Svector::operator*=(const double a) {
  if (a == 0.0) {
    Fs_.clear();
    return *this;
  }
  else if (a == 1.0) {
    return *this;
  }
  else if (a == -1.0) {
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      it->v_ = - it->v_;
    }
    return *this;
  }
  else {
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      if (it->v_ != 0.0)
	it->v_ = a * it->v_;
    }
    return *this;
  }
}

/////////////////////////////////////
// Sparse vector scaling: w /= a
/////////////////////////////////////
Svector& Svector::operator/=(const double a) {
  assert(a != 0.0);
  if (a == 1.0) {
    return *this;
  }
  else if (a == -1.0) {
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      it->v_ = - it->v_;
    }
    return *this;
  }
  else {
    double inv_a = 1.0 / a;
    for (vector<Feature>::iterator it=Fs_.begin(); it<Fs_.end(); it++) {
      if (it->v_ != 0.0)
	it->v_ = it->v_ * inv_a;
    }
    return *this;
  }
}


//////////////////////////////////////////
// Pointwise sparse multiply: w .* = x
//////////////////////////////////////////
Svector& Svector::operator*=(const Svector& x) {
  if (Fs_.empty()) {
    return *this; // w remains unchanged
  }
  else if (x.Fs_.empty()) {
    Fs_.clear();
    return *this;
  }
  else {
    vector<Feature>::iterator itw = Fs_.begin();
    vector<Feature>::const_iterator itx = x.Fs_.begin();
    while (itw < Fs_.end() && itx < x.Fs_.end()) {
      if (itw->i_ == itx->i_) {
        itw->v_ *= itx->v_;
	++itw; ++itx;
      }
      else {
	if (itw->i_ > itx->i_) {
	  ++itx;
	}
	else {
          itw->v_ = 0.0;
	  ++itw;
	}
      }
    }
    return *this;
  }
}

////////////////////////////////////
// Pointwise sparse power: w .^ = p
////////////////////////////////////
Svector& Svector::operator^=(const double p) {
  if (p == 0.0) {
    SetAll(1.0);
    return *this;
  }
  else if (p == 1.0) {
    return *this;
  }
  else if (p == -1.0) {
    for (vector<Feature>::iterator it = Fs_.begin(); it < Fs_.end(); it++) {
      it->v_ = 1.0 / it->v_;
    }
    return *this;
  }
  else if (p == 0.5) {
    for (vector<Feature>::iterator it = Fs_.begin(); it < Fs_.end(); it++) {
      if (it->v_ != 1.0) {
        it->v_ = sqrt(it->v_);
      }
    }
    return *this;
  }
  else {
    for (vector<Feature>::iterator it = Fs_.begin(); it < Fs_.end(); it++) {
      if (it->v_ != 1.0) {
        it->v_ = pow(it->v_, p);
      }
    }
    return *this;
  }
}


/////////////////////////////
// Sparse Dot Product: w^T x
/////////////////////////////
double Svector::SparseDot(const Svector& x) const {
  if (Fs_.empty() || x.Fs_.empty()) {
    return 0.0;
  }
  else {
    double dv = 0.0;
    vector<Feature>::const_iterator itw = Fs_.begin();
    vector<Feature>::const_iterator itx = x.Fs_.begin();
    while (itw < Fs_.end() && itx < x.Fs_.end()) {
      if (itw->i_ == itx->i_) {
	dv += itw->v_ * itx->v_;
	itw++; itx++;
      }
      else {
	if (itw->i_ > itx->i_) {
	  itx++;
	}
	else {
	  itw++;
	}
      }
    }
    return dv;
  }
}

////////////////////////////////////////
// Sparse squared L2 norm: return w^T w
////////////////////////////////////////
double Svector::SparseSqL2Norm() const {
  double v = 0.0;
  for (vector<Feature>::const_iterator it=Fs_.begin(); it<Fs_.end(); it++) {
    if (it->v_ != 0)
      v += pow(it->v_, 2);
  }
  return v;
}

///////////////////////////////////////////
// Sparse vector scaled add: w += a * x
///////////////////////////////////////////
void Svector::SparseAddExpertOverwrite(double a, Svector *x) {
  size_t nz_x = x->Fs_.size();
  size_t ct_w = 0, ct_x = 0;

  if (a == 0.0 || x->Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return;
  }
  else if (Fs_.empty()) { // w: all-0
    Fs_ = x->Fs_;
    for (ct_w = 0; ct_w < Fs_.size(); ct_w ++) {
      Fs_[ct_w].v_ = a * Fs_[ct_w].v_;
    }
    return;
  }
  else { // neither w nor x is of all-0
    while (ct_w<Fs_.size() || ct_x<nz_x) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
	Fs_.push_back( Feature(x->Fs_[ct_x].i_, a * x->Fs_[ct_x].v_) );
	++ct_w; ++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x->Fs_[ct_x].i_) {
	  Fs_[ct_w].v_ += a * x->Fs_[ct_x].v_;
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x->Fs_[ct_x].i_) {
	  Fs_.insert(Fs_.begin()+ct_w, Feature(x->Fs_[ct_x].i_, a * x->Fs_[ct_x].v_));
	  ++ct_w; ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
	  ++ct_w;
	}
      }
    }
  }
}

////////////////////////////////////////
// Sparse vector subtraction: w<= p - n
////////////////////////////////////////
void Svector::SparseSubtract(Svector& p, Svector& n) {
  /*
  *this = p;
  *this -= n;
  */
  size_t nz_w = 0;
  size_t nz_p = p.Fs_.size();
  size_t nz_n = n.Fs_.size();
  size_t ct_p = 0, ct_n = 0, ct_w = 0;

  if (nz_p == 0) {
    if (nz_n == 0) {
      Fs_.clear();
      return;
    }
    else {
      Fs_ = n.Fs_;
      for (ct_n = 0; ct_n < nz_n; ct_n ++) {
        Fs_[ct_n].v_ = - n.Fs_[ct_n].v_;
      }
      return;
    }
  }
  else if (nz_n == 0) {
    Fs_ = p.Fs_;
    return;
  }
  else {
    while (ct_p<nz_p || ct_n<nz_n) {
      if (ct_p == nz_p) {
        ++ct_n;
        ++nz_w;
      }
      else if (ct_n == nz_n) {
        ++ct_p;
        ++nz_w;
      }
      else {
        if (p.Fs_[ct_p].i_ == n.Fs_[ct_n].i_) {
          ++ct_p;
          ++ct_n;
          ++nz_w;
        }
        else if (p.Fs_[ct_p].i_ > n.Fs_[ct_n].i_) {
          ++ct_n;
          ++nz_w;
        }
        else {
          ++ct_p;
          ++nz_w;
        }
      }
    }
    Fs_.resize(nz_w);

    ct_p = 0; ct_n = 0; ct_w = 0;
    while (ct_p<nz_p || ct_n<nz_n) {
      if (ct_p == nz_p) {
        Fs_[ct_w].i_ = n.Fs_[ct_n].i_;
        Fs_[ct_w].v_ = - (n.Fs_[ct_n].v_);
        ++ct_n;
        ++ct_w;
      }
      else if (ct_n == nz_n) {
        Fs_[ct_w].i_ = p.Fs_[ct_p].i_;
        Fs_[ct_w].v_ = p.Fs_[ct_p].v_;
        ++ct_p;
        ++ct_w;
      }
      else {
        if (p.Fs_[ct_p].i_ == n.Fs_[ct_n].i_) {
          Fs_[ct_w].i_ = p.Fs_[ct_p].i_;
          Fs_[ct_w].v_ = p.Fs_[ct_p].v_ - n.Fs_[ct_n].v_;
          ++ct_p;
          ++ct_n;
          ++ct_w;
        }
        else if (p.Fs_[ct_p].i_ > n.Fs_[ct_n].i_) {
          Fs_[ct_w].i_ = n.Fs_[ct_n].i_;
          Fs_[ct_w].v_ = - n.Fs_[ct_n].v_;
          ++ct_n;
          ++ct_w;
        }
        else {
          Fs_[ct_w].i_ = p.Fs_[ct_p].i_;
          Fs_[ct_w].v_ = p.Fs_[ct_p].v_;
          ++ct_p;
          ++ct_w;
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////
// Sparse vector exponential dot multiply: w<= w .* exp(x)
///////////////////////////////////////////////////////////
void Svector::SparseExpMultiplyOverwrite(Svector *x) {
  if (Fs_.size() == 0 || x->Fs_.size() == 0) {
    return; // w remains unchanged
  }
  else { // neither w nor x is of all-0
    vector<Feature>::iterator itw = Fs_.begin();
    vector<Feature>::iterator itx = x->Fs_.begin();
    while (itw < Fs_.end() || itx < x->Fs_.end()) {
      if (itw == Fs_.end() || itx == x->Fs_.end()) { // w/x reaches end
	break;
      }
      else { // neither w nor x reaches end
	if (itw->i_ == itx->i_) {
	  if (itw->v_ != 0 && itx->v_ != 0) {
	    itw->v_ *= exp(itx->v_);
	  }
	  ++itw; ++itx;
	}
	else if(itw->i_ < itx->i_) {
	  ++itw;
	}
	else {
	  ++itx;
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////
// Sparse vector exponential dot multiply: w<= w .* exp(-x)
///////////////////////////////////////////////////////////
void Svector::SparseNegExpMultiplyOverwrite(Svector *x) {
  if (Fs_.empty() || x->Fs_.empty()) {
    return; // w remains unchanged
  }
  else { // neither w nor x is of all-0
    vector<Feature>::iterator itw = Fs_.begin();
    vector<Feature>::iterator itx = x->Fs_.begin();
    while (itw < Fs_.end() || itx < x->Fs_.end()) {
      if (itw == Fs_.end() || itx == x->Fs_.end()) { // w/x reaches end
	break;
      }
      else { // neither w nor x reaches end
	if (itw->i_ == itx->i_) {
	  if (itw->v_ != 0 && itx->v_ != 0) {
	    itw->v_ = itw->v_ / exp(itx->v_);
	  }
	  ++itw; ++itx;
	}
	else if(itw->i_ < itx->i_) {
	  ++itw;
	}
	else {
	  ++itx;
        }
      }
    }
  }
}

//////////////////////////////////////////////////
// Sparse squared Euclidean distance: \|w-x\|_2^2
//////////////////////////////////////////////////
double Svector::SparseSqEuclideanDistance(const Svector& x) const {
  if (x.Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return SparseSqL2Norm();
  }
  else if (Fs_.empty()) { // w: all-0
    return x.SparseSqL2Norm();
  }
  else { // neither w nor x is of all-0
    double d = 0.0;
    size_t ct_w = 0, ct_x = 0;
    while (ct_w<Fs_.size() || ct_x<x.Fs_.size()) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
        d += pow(x.Fs_[ct_x].v_, 2);
	++ct_x;
      }
      else if (ct_x == x.Fs_.size()) { // x reaches end, while w still not
	d += pow(Fs_[ct_w].v_, 2);
        ++ct_w;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x.Fs_[ct_x].i_) {
	  d += pow(Fs_[ct_w].v_ - x.Fs_[ct_x].v_, 2);
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x.Fs_[ct_x].i_) {
          d += pow(x.Fs_[ct_x].v_, 2);
          ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
          d += pow(Fs_[ct_w].v_, 2);
	  ++ct_w;
	}
      }
    }
    return d;
  }
}

///////////////////////////////
// Shrink: remove 0 values
///////////////////////////////
void Svector::Shrink(double threshold) {
  for (size_t i=0; i<Fs_.size(); i++) {
    if (fabs(Fs_[i].v_) < threshold) {
      Fs_.erase(Fs_.begin()+i);
      i--;
    }
  }
}

//---------------------------Example-------------------------//
//////////////////////////
// Create an empty feature
//////////////////////////
Example::Example() : y_(0), in_use_(false), ud_("") {
}

/////////////////////////////////////////////////////////
// Create an unused example from given features and label
////////////////////////////////////////////////////////
Example::Example(const vector<Feature> Fs, const T_LBL y) : 
  Svector(Fs), y_(y), in_use_(false), ud_("") {
}

/////////////////////////////////////////////////////////
// Create an unused example from given features and label
////////////////////////////////////////////////////////
Example::Example(const vector<Feature> Fs, const T_LBL y, const string ud) : 
  Svector(Fs), y_(y), in_use_(false), ud_(ud) {
}

//////////////////////////
// Deconstruct example
//////////////////////////
Example::~Example() {
}

////////////////////////////
// Copy from a given example
////////////////////////////
Example& Example::operator=(const Example& x) {
  Fs_ = x.Fs_;
  y_ = x.y_;
  in_use_ = x.in_use_;
  ud_ = x.ud_;
  return *this;
}

////////////////
// Clear content
////////////////
void Example::Clear() {
  Svector::Clear();
  y_ = 0;
  in_use_ = 0;
  ud_ = "";
}

/////////////////
// Print content
/////////////////
void Example::Print() {
  cout << y_ << " ";
  Svector::Print();
}
