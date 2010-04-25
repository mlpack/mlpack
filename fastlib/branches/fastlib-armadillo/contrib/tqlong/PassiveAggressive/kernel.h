
#ifndef KERNEL_H
#define KERNEL_H

enum KERNEL_TYPE {
  LINEAR_KERNEL, POLYNOMIAL_KERNEL, GAUSSIAN_KERNEL, CUSTOM_KERNEL
};

/** Define kernel function classes
    - operator () to compute kernel
    - linear kernel, polynomial kernel, gaussian kernel
*/
struct KernelFunction {
  KERNEL_TYPE m_eType;
public:
  KernelFunction(KERNEL_TYPE type) { m_eType = type;  }
  virtual double operator()(const Vector&, const Vector&) = 0;
  virtual ~KernelFunction() {}
};

struct LinearKernel : KernelFunction {
public:
  LinearKernel() : KernelFunction(LINEAR_KERNEL) {}
  double operator()(const Vector&, const Vector&);
};

struct PolynomialKernel : KernelFunction {
  index_t m_iPolyOrder;
  bool m_bHomogeneous;
public:
  PolynomialKernel(index_t order, bool homogeneous = true) :
    KernelFunction(POLYNOMIAL_KERNEL) {
    m_iPolyOrder = order;
    m_bHomogeneous = homogeneous;
  }
  double operator()(const Vector&, const Vector&);
};

struct Gaussian2Kernel : KernelFunction {
  double m_dSigma2;
public:
  Gaussian2Kernel(double sigma) : KernelFunction(GAUSSIAN_KERNEL) {
    m_dSigma2 = sigma*sigma;
  }
  double operator()(const Vector&, const Vector&);
};

#endif
