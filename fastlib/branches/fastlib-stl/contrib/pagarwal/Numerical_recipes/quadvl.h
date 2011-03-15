void quadvl(const Doub x, const Doub y, Doub &fa, Doub &fb, Doub &fc, Doub &fd)
{
	Doub qa,qb,qc,qd;
	qa=MIN(2.0,MAX(0.0,1.0-x));
	qb=MIN(2.0,MAX(0.0,1.0-y));
	qc=MIN(2.0,MAX(0.0,x+1.0));
	qd=MIN(2.0,MAX(0.0,y+1.0));
	fa=0.25*qa*qb;
	fb=0.25*qb*qc;
	fc=0.25*qc*qd;
	fd=0.25*qd*qa;
}
