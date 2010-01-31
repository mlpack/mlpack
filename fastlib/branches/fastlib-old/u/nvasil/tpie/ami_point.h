// Copyright (C) 2002 Octavian Procopiuc
//
// File:    ami_point.h
// Author:  Octavian Procopiuc <tavi@cs.duke.edu>
//
// The AMI_point and AMI_record classes.
//
// $Id: ami_point.h,v 1.9 2005/01/21 16:55:29 tavi Exp $
//

#ifndef AMI_POINT_H_
#define AMI_POINT_H_

// For ostream.
#include <iostream>

// This is a hack. It works for integer types only.
template<class coord_t>
class infinity_t {
  static coord_t minf;// = (1 << (8*sizeof(coord_t) - 1));
  static coord_t pinf;// = ~minf;
public:
  int operator+() const { return pinf; }
  int operator-() const { return minf; }
};

template<class coord_t>
coord_t infinity_t<coord_t>::minf = (1 << (8*sizeof(coord_t) - 1));
template<class coord_t>
coord_t infinity_t<coord_t>::pinf = ~(1 << (8*sizeof(coord_t) - 1));

//int infinity_t<int>::minf = (1 << (8*sizeof(int) - 1));
//int infinity_t<int>::pinf = ~(1 << (8*sizeof(int) - 1));

// The base class for AMI_point.
template<class coord_t, size_t dim>
class AMI_point_base {
protected:
  coord_t coords_[dim];
public:
  static infinity_t<coord_t> Inf;

  // The default constructor.
  AMI_point_base() {}

  // The array operators for accessing/setting the coordinates.
  coord_t& operator[](size_t i) { return coords_[i]; }
  const coord_t& operator[](size_t i) const { return coords_[i]; }


  // Operator < for window queries. It's actually more like <=.
  bool operator<(const AMI_point_base<coord_t, dim>& p) const {
    size_t i;
    for (i = 0; i < dim; i++)
      if (p[i] < coords_[i])
	break;
    return (i == dim);
  }

  void set_min(const AMI_point_base<coord_t, dim>& p) {
    for (size_t j = 0; j < dim; j++)
      coords_[j] = min(coords_[j], p[j]);
  }

  void set_max(const AMI_point_base<coord_t, dim>& p) {
    for (size_t j = 0; j < dim; j++)
      coords_[j] = max(coords_[j], p[j]);
  }

  // Scalar product.
  coord_t operator*(const AMI_point_base<coord_t,dim>& p) const {
    coord_t ans = coords_[0]*p[0];
    for (size_t i = 1; i < dim; i++)
      ans += coords_[i]*p[i];
    return ans;
  }
};

template<class coord_t, size_t dim>
infinity_t<coord_t> AMI_point_base<coord_t, dim>::Inf = infinity_t<coord_t>();


// The AMI_point class.
template <class coord_t, size_t dim>
class AMI_point: public AMI_point_base<coord_t, dim> {
 protected:
  using AMI_point_base<coord_t, dim>::coords_;
  
 public:
  bool operator==(const AMI_point<coord_t, dim>& p) const {
    size_t i = 0;
    while (i < dim) {
      if (coords_[i] != p[i])
	break;
      i++;
    }
    return (i == dim);
  }

  // The comparison class. For sorting on each of the dim dimensions.
  class cmp {
    // The dimension on which to compare. It should be less than dim.
    size_t d_;
  public:
    cmp(size_t d = 0): d_(d) {}

    inline int compare(const AMI_point<coord_t,dim>& p1, 
                const AMI_point<coord_t,dim>& p2) const {
      // Lexicographic order starting with dimension d_.
      if (p1[d_] < p2[d_])
        return -1;
      else if (p1[d_] > p2[d_])
        return 1;
      else 
	return _compare(p1, p2);
    }
    
    // This operator is used by STL sort().
    bool operator()(const AMI_point<coord_t, dim>& p1, 
		    const AMI_point<coord_t, dim>& p2) const {
      return (compare(p1, p2) == -1);
    }

  private:
    int _compare(const AMI_point<coord_t,dim>& p1, 
		 const AMI_point<coord_t,dim>& p2) const {
      size_t j = 0;
      // Cycle once through all dimensions, starting with d_.
      while (j < dim && p1[(j+d_)%dim] == p2[(j+d_)%dim])
	j++;
      if (j == dim)
	return 0;
      else
	return (p1[(j+d_)%dim] < p2[(j+d_)%dim]) ? -1: 1;
    }
  };
};

template<class coord_t, size_t dim>
ostream& operator<<(ostream& s, const AMI_point<coord_t, dim>& p) {
  for (size_t i = 0; i < dim-1; i++)
    s << p[i] << " ";
  return s << p[dim-1];
}

#ifdef _WIN32

#else
// Partial specialization of AMI_point for 2 dimensions.
template <class coord_t>
class AMI_point<coord_t, 2>: public AMI_point_base<coord_t, 2> {
 protected:
  using AMI_point_base<coord_t, 2>::coords_;
  
 public:

  AMI_point() {}
  AMI_point(coord_t _x, coord_t _y) { coords_[0] = _x; coords_[1] = _y; }
  coord_t x() const { return coords_[0]; }
  coord_t& x() { return coords_[0]; }
  coord_t y() const { return coords_[1]; }
  coord_t& y() { return coords_[1]; }

  bool operator==(const AMI_point<coord_t, 2>& p) const {
    return (coords_[0] == p.coords_[0]) && 
      (coords_[1] == p.coords_[1]); 
  }
  bool operator!=(const AMI_point<coord_t, 2>& p) const {
    return (coords_[0] != p.coords_[0]) || 
      (coords_[1] != p.coords_[1]); 
  }
  bool less_x(const AMI_point<coord_t, 2>& b) const {
    return (coords_[0] < b.coords_[0]) || 
      ((coords_[0] == b.coords_[0]) && (coords_[1] < b.coords_[1]));
  }
  bool less_y(const AMI_point<coord_t, 2>& b) const { 
    return (coords_[1] < b.coords_[1]) || 
      ((coords_[1] == b.coords_[1]) && (coords_[0] < b.coords_[0]));
  }
  struct less_X {
    bool operator()(const AMI_point<coord_t, 2>& a, 
			   const AMI_point<coord_t, 2>& b) const 
    { return (a[0] < b[0]) || ((a[0] == b[0]) && (a[1] < b[1])); }
  };

  struct less_Y {
    bool operator()(const AMI_point<coord_t, 2>& a, 
			   const AMI_point<coord_t, 2>& b) const 
    { return (a[1] < b[1]) || ((a[1] == b[1]) && (a[0] < b[0])); }
  };

  // The comparison class. For sorting on each of the dim dimensions.
  class cmp {
    // The dimension on which to compare. It should be less than 2.
    size_t d_;
  public:
    cmp(size_t d = 0): d_(d) {}

    inline int compare(const AMI_point<coord_t, 2>& p1, 
                const AMI_point<coord_t, 2>& p2) const {
      // Lexicographic order starting with dimension d_.
      if (p1[d_] < p2[d_])
        return -1;
      else if (p1[d_] > p2[d_])
        return 1;
      else if (p1[(d_+1)%2] < p2[(d_+1)%2])
	return -1;
      else if (p2[(d_+1)%2] < p1[(d_+1)%2])
	return 1;
      else
	return 0;
    }
    
    // This operator is used by STL sort().
    bool operator()(const AMI_point<coord_t, 2>& p1, 
		    const AMI_point<coord_t, 2>& p2) const {
      return (compare(p1, p2) == -1);
    }
  };
};

template<class coord_t>
ostream& operator<<(ostream& s, const AMI_point<coord_t, 2>& p) {
  return s << p[0] << " " << p[1];
}
#endif // !_WIN32


template<class coord_t, class data_t, size_t dim>
class AMI_record_base {
public:
  typedef AMI_point<coord_t, dim> point_t;

  point_t key;
  data_t data;

  //  AMI_record() {}
  AMI_record_base(const point_t& p, data_t _data = data_t(0)): 
    key(p), data(_data) {}
  AMI_record_base(data_t _data = data_t(0)): data(_data) {}

  data_t& id() { return data; }
  const data_t& id() const { return data; }

  bool operator==(const AMI_record_base<coord_t, data_t, dim>& r) const
  { return key == r.key; }

  // The array operators for accessing/setting the coordinates.
  coord_t& operator[](size_t i) { return key[i]; }
  const coord_t& operator[](size_t i) const { return key[i]; }

   // Operator < for window queries. It's actually more like <=.
  bool operator<(const AMI_record_base<coord_t, data_t, dim>& p) const {
    size_t i;
    for (i = 0; i < dim; i++)
      if (p[i] < key[i])
	break;
    return (i == dim);
  }


  void set_min(const AMI_record_base<coord_t, data_t, dim>& p) {
    for (size_t j = 0; j < dim; j++)
      key[j] = min(key[j], p[j]);
  }

  void set_max(const AMI_record_base<coord_t, data_t, dim>& p) {
    for (size_t j = 0; j < dim; j++)
      key[j] = max(key[j], p[j]);
  }

  // Scalar product.
  coord_t operator*(const AMI_record_base<coord_t, data_t, dim>& p) const {
    coord_t ans = key[0] * p[0];
    for (size_t i = 1; i < dim; i++)
      ans += key[i] * p[i];
    return ans;
  } 
};


template<class coord_t, class data_t, size_t dim>
class AMI_record: public AMI_record_base<coord_t, data_t, dim> {
public:
  AMI_record(const typename AMI_record_base<coord_t, data_t, dim>::point_t& p, data_t b = data_t(0)): 
    AMI_record_base<coord_t, data_t, dim>(p, b) {}
  AMI_record(data_t b = data_t(0)): AMI_record_base<coord_t, data_t, dim>(b) {}

  // The comparison class. For sorting on each of the dim dimensions.
  class cmp {
    // The dimension on which to compare. It should be less than dim.
    size_t d_;
  public:
    cmp(size_t d = 0): d_(d) {}

    inline int compare(const AMI_record<coord_t, data_t, dim>& p1, 
                const AMI_record<coord_t, data_t, dim>& p2) const {
      // Lexicographic order starting with dimension d_.
      if (p1[d_] < p2[d_])
        return -1;
      else if (p2[d_] < p1[d_])
        return 1;
      else 
	return _compare(p1, p2);
    }
    
    // This operator is used by STL sort().
    bool operator()(const AMI_record<coord_t, data_t, dim>& p1, 
		    const AMI_record<coord_t, data_t, dim>& p2) const {
      return (compare(p1, p2) == -1);
    }

  private:
    int _compare(const AMI_record<coord_t, data_t, dim>& p1, 
		 const AMI_record<coord_t, data_t, dim>& p2) const {
      size_t j = 0;
      // Cycle once through all dimensions, starting with d_.
      // TODO: change equality expression to an expression containing only <.
      while (j < dim && p1[(j+d_)%dim] == p2[(j+d_)%dim])
	j++;
      if (j == dim)
	return 0;
      else
	return (p1[(j+d_)%dim] < p2[(j+d_)%dim]) ? -1: 1;
    }
  };

};

template<class coord_t, class data_t, size_t dim>
ostream& operator<<(ostream& s, const AMI_record<coord_t, data_t, dim>& p) {
  for (TPIE_OS_TIME_T i = 0; i < dim; i++)
    s << p[i] << " ";
  return s << (TPIE_OS_OFFSET)p.id();
}

#ifdef _WIN32

#else
// A record consists of a key (two-dimensional point) and a data
// item. Used by the EPStree.
template<class coord_t, class data_t>
struct AMI_record<coord_t, data_t, 2>: public AMI_record_base<coord_t, data_t, 2> {
public:
  AMI_record(const typename AMI_record_base<coord_t, data_t, 2>::point_t& p, data_t b = data_t(0)): 
    AMI_record_base<coord_t, data_t, 2>(p, b) {}
  AMI_record(const coord_t& x, const coord_t& y, const data_t& b): 
    AMI_record_base<coord_t, data_t, 2>(point_t(x, y), b) {}
  AMI_record(data_t b = data_t(0)): AMI_record_base<coord_t, data_t, 2>(b) {}

  struct less_X_point {
    bool operator()(const AMI_record<coord_t, data_t, 2>& r, 
			   const typename AMI_record_base<coord_t, data_t, 2>::point_t& p) const { 
      //      return point_t::less_X()(r.key, p); 
      return r.key.less_x(p);
    }
  };

  struct less_X {
    bool operator()(const AMI_record<coord_t, data_t, 2>& r1, 
			   const AMI_record<coord_t, data_t, 2>& r2) const {
      //      return point_t::less_X()(r1.key, r2.key);
      return r1.key.less_x(r2.key);
    }

    int compare(const AMI_record<coord_t, data_t, 2>& r1, 
		const AMI_record<coord_t, data_t, 2>& r2) const {
      if (r1.key.less_x(r2.key))
	return -1;
      else if (r2.key.less_x(r1.key))
	return 1;
      else
	return 0;
    }
  };

  struct less_Y {
    bool operator()(const AMI_record<coord_t, data_t, 2>& r1, 
			   const AMI_record<coord_t, data_t, 2>& r2) const {
      //      return point_t::less_Y()(r1.key, r2.key);
      return r1.key.less_y(r2.key);
    }

    int compare(const AMI_record<coord_t, data_t, 2>& r1, 
		const AMI_record<coord_t, data_t, 2>& r2) const {
      if (r1.key.less_y(r2.key))
	return -1;
      else if (r2.key.less_y(r1.key))
	return 1;
      else
	return 0;
    }
  };

  // The comparison class. For sorting on each of the dim dimensions.
  class cmp {
    // The dimension on which to compare. It should be less than 2.
    size_t d_;
  public:
    cmp(size_t d = 0): d_(d) {}

    inline int compare(const AMI_record<coord_t, data_t, 2>& p1, 
                const AMI_record<coord_t, data_t, 2>& p2) const {
      // Lexicographic order starting with dimension d_.
      if (p1[d_] < p2[d_])
        return -1;
      else if (p2[d_] < p1[d_])
        return 1;
      else if (p1[(d_+1)%2] < p2[(d_+1)%2])
	return -1;
      else if (p2[(d_+1)%2] < p1[(d_+1)%2])
	return 1;
      else
	return 0;
    }
    
    // This operator is used by STL sort().
    bool operator()(const AMI_record<coord_t, data_t, 2>& p1, 
		    const AMI_record<coord_t, data_t, 2>& p2) const {
      return (compare(p1, p2) == -1);
    }
  };
};

template<class coord_t, class data_t>
ostream& operator<<(ostream& s, const AMI_record<coord_t, data_t, 2>& p) {
  return s << p[0] << " " << p[1] << " " << p.id();
}
#endif // !_WIN32


// Function object to extract the key from a record.
template<class coord_t, class data_t, size_t dim>
class AMI_record_key {
public:
  AMI_point<coord_t, dim> operator()(const AMI_record<coord_t, data_t, dim>& r) const
  { return r.key; }
};

#endif // AMI_POINT_H_
