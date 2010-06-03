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

  void operator=(const Transformation& t) {
    m.CopyValues(t.m);
  }

  void Print() const {
    printf("%8f %8f %8f\n"
	   "%8f %8f %8f\n"
	   "%8f %8f %8f\n", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], 1.0);
  }

  void Print(FILE* f) const {
    fprintf(f, "%8f %8f %8f\n"
	    "%8f %8f %8f\n"
	    "%8f %8f %8f\n"
	    "--------------\n", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], 1.0);
  }
};

struct PointType {
  double r;    // row
  double c;    // col
  double f;    // feature value or intensity 

  PointType(double r_ = 0, double c_ = 0, double f_ = 0) { 
    r = r_; c = c_; f = f_;
  }

  void setValue(double r_, double c_, double f_) { 
    r = r_; c = c_; f = (f_>=0) ? f_ : 0; // f cannot be negative
  }

  void AddNumeric(const PointType& p, double lambda = 1.0) {
    setValue(r+p.r*lambda, c+p.c*lambda, f+p.f*lambda);
  }

  PointType Transform(const Transformation& t, double s = 1.0) const;
};

double exp_kernel(const PointType&, const PointType&);
double d_exp_kernel(const PointType& p1, const PointType& p2,
		    double& dr2, double& dc2, double& df2);

struct ImageType {
  ArrayList<PointType> pList;
  
  // Members
  ImageType() {
    pList.Init();
  }

  ImageType(const ImageType& image) {
    pList.InitCopy(image.pList);
  }
  
  ImageType(const char* filename);
  
  ImageType(index_t n_points); // Random initialization;

  void Save(const char* filename) const;
  void Save(FILE* f) const;

  void Add(const ImageType& image, double w = 1.0) {
    index_t old_n = n_points();
    pList.AppendCopy(image.pList);
    if (w != 1.0) 
      for (index_t i = old_n; i < n_points(); i++) pList[i].f *= w;
  }

  double Difference(const ImageType& image, 
		    double (*kernel)(const PointType&, 
				     const PointType&) = exp_kernel) const;
  
  void Scale(ImageType& image_out, double s = 1.0) const;
  
  void Transform(ImageType& image_out, const Transformation& t, double s = 1.0) const;
  
  index_t n_points() const { return pList.size(); }
};

void Save(FILE* f, const char* name, const ArrayList<ImageType>& X);
void Save(FILE* f, const char* name, const ArrayList<Transformation>& T);
void Save(FILE* f, const char* name, const ArrayList<Vector>& W);
void LoadImageList(ArrayList<ImageType>& B, const char** fn, size_t n_bases);
void RandomImageList(ArrayList<ImageType>& B, size_t n_bases, index_t n_points);
