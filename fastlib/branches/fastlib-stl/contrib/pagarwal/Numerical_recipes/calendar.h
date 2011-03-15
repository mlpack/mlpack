Int julday(const Int mm, const Int id, const Int iyyy) {
	const Int IGREG=15+31*(10+12*1582);
	Int ja,jul,jy=iyyy,jm;
	if (jy == 0) throw("julday: there is no year zero.");
	if (jy < 0) ++jy;
	if (mm > 2) {
		jm=mm+1;
	} else {
		--jy;
		jm=mm+13;
	}
	jul = Int(floor(365.25*jy)+floor(30.6001*jm)+id+1720995);
	if (id+31*(mm+12*iyyy) >= IGREG) {
		ja=Int(0.01*jy);
		jul += 2-ja+Int(0.25*ja);
	}
	return jul;
}
void caldat(const Int julian, Int &mm, Int &id, Int &iyyy) {
	const Int IGREG=2299161;
	Int ja,jalpha,jb,jc,jd,je;

	if (julian >= IGREG) {
		jalpha=Int((Doub(julian-1867216)-0.25)/36524.25);
		ja=julian+1+jalpha-Int(0.25*jalpha);
	} else if (julian < 0) {
		ja=julian+36525*(1-julian/36525);
	} else
		ja=julian;
	jb=ja+1524;
	jc=Int(6680.0+(Doub(jb-2439870)-122.1)/365.25);
	jd=Int(365*jc+(0.25*jc));
	je=Int((jb-jd)/30.6001);
	id=jb-jd-Int(30.6001*je);
	mm=je-1;
	if (mm > 12) mm -= 12;
	iyyy=jc-4715;
	if (mm > 2) --iyyy;
	if (iyyy <= 0) --iyyy;
	if (julian < 0) iyyy -= 100*(1-julian/36525);
}
void flmoon(const Int n, const Int nph, Int &jd, Doub &frac) {
	const Doub RAD=3.141592653589793238/180.0;
	Int i;
	Doub am,as,c,t,t2,xtra;
	c=n+nph/4.0;
	t=c/1236.85;
	t2=t*t;
	as=359.2242+29.105356*c;
	am=306.0253+385.816918*c+0.010730*t2;
	jd=2415020+28*n+7*nph;
	xtra=0.75933+1.53058868*c+((1.178e-4)-(1.55e-7)*t)*t2;
	if (nph == 0 || nph == 2)
		xtra += (0.1734-3.93e-4*t)*sin(RAD*as)-0.4068*sin(RAD*am);
	else if (nph == 1 || nph == 3)
		xtra += (0.1721-4.0e-4*t)*sin(RAD*as)-0.6280*sin(RAD*am);
	else throw("nph is unknown in flmoon");
	i=Int(xtra >= 0.0 ? floor(xtra) : ceil(xtra-1.0));
	jd += i;
	frac=xtra-i;
}
