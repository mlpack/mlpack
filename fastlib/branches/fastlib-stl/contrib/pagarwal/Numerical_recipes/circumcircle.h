struct Circle {
	Point<2> center;
	Doub radius;
	Circle(const Point<2> &cen, Doub rad) : center(cen), radius(rad) {}
};

Circle circumcircle(Point<2> a, Point<2> b, Point<2> c) {
	Doub a0,a1,c0,c1,det,asq,csq,ctr0,ctr1,rad2;
	a0 = a.x[0] - b.x[0]; a1 = a.x[1] - b.x[1];
	c0 = c.x[0] - b.x[0]; c1 = c.x[1] - b.x[1];
	det = a0*c1 - c0*a1;
	if (det == 0.0) throw("no circle thru colinear points");
	det = 0.5/det;
	asq = a0*a0 + a1*a1;
	csq = c0*c0 + c1*c1;
	ctr0 = det*(asq*c1 - csq*a1);
	ctr1 = det*(csq*a0 - asq*c0);
	rad2 = ctr0*ctr0 + ctr1*ctr1;
	return Circle(Point<2>(ctr0 + b.x[0], ctr1 + b.x[1]), sqrt(rad2));
}
