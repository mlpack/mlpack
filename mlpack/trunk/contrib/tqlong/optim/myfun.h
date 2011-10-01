
#pragma once

class MyFun {
  size_t dim;
public:
  MyFun() : dim(2) { }

  size_t n_dim() {
    return dim;
  }

  size_t n_con() {
    return dim;
  }

  size_t n_eq() {
    return 0;
  }

  double fValue(const Vector& x) { // sum x[i]^2
    for (int i = 0; i < x.length(); i++)
      if (cValue(i, x) > 0) return INFINITY;
    return la::Dot(x, x);
  }

  void fGradient(const Vector& x, Vector* g) { // g = 2x
    la::ScaleOverwrite(2, x, g);
  }
  
  double cValue(size_t i, const Vector& x) {
    DEBUG_ASSERT(i < n_con());
    return -0.4*(i+1)-x[i];
  }

  void cGradient(size_t i, const Vector& x, Vector* gc) {
    DEBUG_ASSERT(i < n_con());
    gc->SetAll(0.0);
    (*gc)[i] = -1.0;
  }

  double eqValue(size_t i, const Vector& x) {
    DEBUG_ASSERT(i < n_eq());
    return INFINITY;
  }

  void eqGradient(size_t i, const Vector& x, Vector* gc) {
    DEBUG_ASSERT(i < n_eq());
  }

  double Dot(const Vector& x, const Vector& y) {
    return la::Dot(x, y);
  }

  void AddExpert(double alpha, const Vector& x, Vector* y) {
    la::AddExpert(alpha, x, y);
  }

  void SubOverwrite(const Vector &x, const Vector& y, Vector* z) {
    la::SubOverwrite(x, y, z);
  }

  void Scale(double alpha, Vector* x) {
    la::Scale(alpha, x);
  }

  void ScaleOverwrite(double alpha, const Vector& x, Vector* y) {
    la::ScaleOverwrite(alpha, x, y);
  }

  void MulOverwrite(const Matrix &A, const Vector& x, Vector* y) {
    la::MulOverwrite(A, x, y);
  }

  void MulOverwrite(const Vector& x, const Matrix &A, Vector* y) {
    la::MulOverwrite(x, A, y);
  }
};

class MyFun1 {
  size_t dim;
public:
  MyFun1(size_t d) : dim(d) { }

  size_t n_dim() {
    return dim;
  }

  size_t n_con() {
    return dim;
  }

  size_t n_eq() {
    return 0;
  }

  double fValue(const Vector& x) { // sum x[i]^2
    for (int i = 0; i < x.length(); i++)
      if (cValue(i, x) > 0) return INFINITY;
    double s = 0;
    for (int i = 0; i < x.length(); i++)
      s += exp(x[i]);
    return log(s);
  }

  void fGradient(const Vector& x, Vector* g) { // g = 2x
    double s = 0;
    for (int i = 0; i < x.length(); i++)
      s += exp(x[i]);
    for (int i = 0; i < x.length(); i++)
      (*g)[i] = exp(x[i])/s;
  }
  
  double cValue(size_t i, const Vector& x) {
    DEBUG_ASSERT(i < n_con());
    return -0.4*(i+1)-x[i];
  }

  void cGradient(size_t i, const Vector& x, Vector* gc) {
    DEBUG_ASSERT(i < n_con());
    gc->SetAll(0.0);
    (*gc)[i] = -1.0;
  }

  double eqValue(size_t i, const Vector& x) {
    DEBUG_ASSERT(i < n_eq());
    return INFINITY;
  }

  void eqGradient(size_t i, const Vector& x, Vector* gc) {
    DEBUG_ASSERT(i < n_eq());
  }

  double Dot(const Vector& x, const Vector& y) {
    return la::Dot(x, y);
  }

  void AddExpert(double alpha, const Vector& x, Vector* y) {
    la::AddExpert(alpha, x, y);
  }

  void SubOverwrite(const Vector &x, const Vector& y, Vector* z) {
    la::SubOverwrite(x, y, z);
  }

  void Scale(double alpha, Vector* x) {
    la::Scale(alpha, x);
  }

  void ScaleOverwrite(double alpha, const Vector& x, Vector* y) {
    la::ScaleOverwrite(alpha, x, y);
  }

  void MulOverwrite(const Matrix &A, const Vector& x, Vector* y) {
    la::MulOverwrite(A, x, y);
  }

  void MulOverwrite(const Vector& x, const Matrix &A, Vector* y) {
    la::MulOverwrite(x, A, y);
  }
};

