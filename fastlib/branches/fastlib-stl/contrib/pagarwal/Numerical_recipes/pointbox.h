template<Int DIM> struct Point {
	Doub x[DIM];
	Point(const Point &p) {
		for (Int i=0; i<DIM; i++) x[i] = p.x[i];
	}	
	Point& operator= (const Point &p) {
		for (Int i=0; i<DIM; i++) x[i] = p.x[i];
		return *this;
	}
	bool operator== (const Point &p) const {
		for (Int i=0; i<DIM; i++) if (x[i] != p.x[i]) return false;
		return true;
	}
	Point(Doub x0 = 0.0, Doub x1 = 0.0, Doub x2 = 0.0) {
		x[0] = x0;
		if (DIM > 1) x[1] = x1;
		if (DIM > 2) x[2] = x2;
		if (DIM > 3) throw("Point not implemented for DIM > 3");
	}
};
template<Int DIM> struct Box {
	Point<DIM> lo, hi;
	Box() {}
	Box(const Point<DIM> &mylo, const Point<DIM> &myhi) : lo(mylo), hi(myhi) {}
};
template<Int DIM> Doub dist(const Point<DIM> &p, const Point<DIM> &q) {
	Doub dd = 0.0;
	for (Int j=0; j<DIM; j++) dd += SQR(q.x[j]-p.x[j]);
	return sqrt(dd);
}
template<Int DIM> Doub dist(const Box<DIM> &b, const Point<DIM> &p) {
	Doub dd = 0;
	for (Int i=0; i<DIM; i++) {
		if (p.x[i]<b.lo.x[i]) dd += SQR(p.x[i]-b.lo.x[i]);
		if (p.x[i]>b.hi.x[i]) dd += SQR(p.x[i]-b.hi.x[i]);
	}
	return sqrt(dd);
}
