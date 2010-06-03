
#include <fastlib/fastlib.h>
#include "image_type.h"

ImageType::ImageType(const char* filename) {
  Matrix tmp;
  data::Load(filename, &tmp);
  pList.Init(tmp.n_cols());
  for (index_t i = 0; i < tmp.n_cols(); i++)
    pList[i].setValue(tmp.get(0, i), tmp.get(1, i), tmp.get(2, i));
}

double ImageType::Difference(const ImageType& image, 
			     double (*kernel)(const PointType&, 
					      const PointType&) ) const {
  double s = 0;
  for (index_t i = 0; i < pList.size(); i++)
    for (index_t j = 0; j < image.pList.size(); j++)
      s += kernel(pList[i], image.pList[j]);
  return s;
}

double exp_kernel(const PointType& p1, const PointType& p2) {
  double sigma2 = fx_param_double(NULL, "sigma", 0.5);
  double gamma = fx_param_double(NULL, "gamma", 0.1);
  double ds = (p1.r-p2.r)*(p1.r-p2.r) + (p1.c-p2.c)*(p1.c-p2.c);
  double df = (p1.f-p2.f)*(p1.f-p2.f);
  //printf("ds = %f df = %f\n", ds, df);
  return -exp(-(sigma2*ds + gamma*df));
}

double d_exp_kernel(const PointType& p1, const PointType& p2,
		    double& dr2, double& dc2, double& df2) {
  double sigma2 = fx_param_double(NULL, "sigma", 0.5);
  double gamma = fx_param_double(NULL, "gamma", 0.1);
  double ds = (p1.r-p2.r)*(p1.r-p2.r) + (p1.c-p2.c)*(p1.c-p2.c);
  double df = (p1.f-p2.f)*(p1.f-p2.f);
  //printf("ds = %f df = %f\n", ds, df);
  double retval =  -exp(-(sigma2*ds + gamma*df));
  dr2 = retval * (-2)*sigma2*(p2.r-p1.r);
  dc2 = retval * (-2)*sigma2*(p2.c-p1.c);
  df2 = retval * (-2)*gamma*(p2.f-p1.f);  
  return retval;
}

void ImageType::Scale(ImageType& image_out, double s) const {
  image_out.pList.Resize(pList.size());
  for (index_t i = 0; i < pList.size(); i++)
    image_out.pList[i].setValue(pList[i].r, pList[i].c, pList[i].f * s);
}

void ImageType::Transform(ImageType& image_out, 
			  const Transformation& t, double s) const {
  image_out.pList.Resize(pList.size());
  for (index_t i = 0; i < pList.size(); i++)
    image_out.pList[i] = pList[i].Transform(t, s);
}

void ImageType::Save(const char* filename) const {
  FILE* f = fopen(filename, "w");
  Save(f);
  fclose(f);
}

void ImageType::Save(FILE* f) const {
  for (index_t i = 0; i < pList.size(); i++) 
    fprintf(f, "%g %g %g\n", pList[i].r, pList[i].c, pList[i].f);
  fprintf(f, "--------------\n");
}

PointType PointType::Transform(const Transformation& t, double s) const {
  PointType newPoint;
  double d = t.m[6]*r + t.m[7]*c + 1;
  newPoint.r = (t.m[0]*r + t.m[1]*c + t.m[2])/d;
  newPoint.c = (t.m[3]*r + t.m[4]*c + t.m[5])/d;
  newPoint.f = f * s;
  return newPoint;
}
  
