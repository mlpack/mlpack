struct NRf3 {
	Doub xsav,ysav;
	Doub (*func3d)(const Doub, const Doub, const Doub);
	Doub operator()(const Doub z)
	{
		return func3d(xsav,ysav,z);
	}
};
struct NRf2 {
	NRf3 f3;
	Doub (*z1)(Doub, Doub);
	Doub (*z2)(Doub, Doub);
	NRf2(Doub zz1(Doub, Doub), Doub zz2(Doub, Doub)) : z1(zz1), z2(zz2) {}
	Doub operator()(const Doub y)
	{
		f3.ysav=y;
		return qgaus(f3,z1(f3.xsav,y),z2(f3.xsav,y));
	}
};
struct NRf1 {
	Doub (*y1)(Doub);
	Doub (*y2)(Doub);
	NRf2 f2;
	NRf1(Doub yy1(Doub), Doub yy2(Doub), Doub z1(Doub, Doub),
		Doub z2(Doub, Doub)) : y1(yy1),y2(yy2), f2(z1,z2) {}
	Doub operator()(const Doub x)
	{
		f2.f3.xsav=x;
		return qgaus(f2,y1(x),y2(x));
	}
};

template <class T>
Doub quad3d(T &func, const Doub x1, const Doub x2, Doub y1(Doub), Doub y2(Doub),
	Doub z1(Doub, Doub), Doub z2(Doub, Doub))
{
	NRf1 f1(y1,y2,z1,z2);
	f1.f2.f3.func3d=func;
	return qgaus(f1,x1,x2);
}
