template<Int DIM> struct Sphcirc {
	Point<DIM> center;
	Doub radius;
	Sphcirc() {}
	Sphcirc(const Point<DIM> &mycenter, Doub myradius)
		: center(mycenter), radius(myradius) {}
	bool operator== (const Sphcirc &s) const {
		return (radius == s.radius && center == s.center);
	}
	Int isinbox(const Box<DIM> &box) {
		for (Int i=0; i<DIM; i++) {
			if ((center.x[i] - radius < box.lo.x[i]) ||
				(center.x[i] + radius > box.hi.x[i])) return 0;
		}
		return 1;
	}
	Int contains(const Point<DIM> &point) {
		if (dist(point,center) > radius) return 0;
		else return 1;	
	}
	Int collides(const Sphcirc<DIM> &circ) {
		if (dist(circ.center,center) > circ.radius+radius) return 0;
		else return 1;
	}
};
