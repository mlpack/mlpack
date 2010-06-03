
#include <fastlib/fastlib.h>
#include "image_type.h"

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
  double sigma2 = 1;
  double s = (p1.r-p2.r)*(p1.r-p2.r) + (p1.c-p2.c)*(p1.c-p2.c);
  return exp(-0.5/sigma2*s) * (p1.f-p2.f)*(p1.f-p2.f);
}

void ImageType::Scale(ImageType& image_out, double s) {
  image_out.pList.Renew();
  image_out.pList.InitCopy(pList);
  for (index_t i = 0; i < pList.size(); i++)
    image_out.pList[i].f *= s;
}

void ImageType::Transform(ImageType& image_out, 
			  const Transformation& t, double s) {
  image_out.pList.Renew();
  image_out.pList.Init(pList.size());
  for (index_t i = 0; i < pList.size(); i++)
    image_out.pList[i] = pList[i].Transform(t, s);
}
