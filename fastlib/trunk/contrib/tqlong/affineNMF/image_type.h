#pragma once

/** image_type.h 
 **/

struct Transformation {
  Vector m;
  Transformation() {
    m.Init(8);
    m[0] = 1; m[1] = 0; m[2] = 0;
    m[3] = 0; m[4] = 1; m[5] = 0;
    m[6] = 0; m[7] = 0;
  }
  Transformation(const Transformation& t) {
    m.Copy(t.m);
  }
};

struct PointType {
  double r;    // row
  double c;    // col
  double f;    // feature value or intensity 

  PointType Transform(const Transformation& t, double s = 1.0) const {
    PointType newPoint;
    double d = t.m[6]*r + t.m[7]*c + 1;
    newPoint.r = (t.m[0]*r + t.m[1]*c + t.m[2])/d;
    newPoint.c = (t.m[3]*r + t.m[4]*c + t.m[5])/d;
    newPoint.f = f * s;
    return newPoint;
  }
};

double exp_kernel(const PointType&, const PointType&);

struct ImageType {
  ArrayList<PointType> pList;
  
  // Members
  ImageType() {
    pList.Init();
  }

  ImageType(const ImageType& image) {
    pList.InitCopy(image.pList);
  }

  void Add(const ImageType& image) {
    pList.AppendCopy(image.pList);
  }

  double Difference(const ImageType& image, 
		    double (*kernel)(const PointType&, 
				     const PointType&) = exp_kernel) const;
  
  void Scale(ImageType& image_out, double s = 1.0);
  
  void Transform(ImageType& image_out, const Transformation& t, double s = 1.0);
  
};

