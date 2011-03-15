void ksone(VecDoub_IO &data, Doub func(const Doub), Doub &d, Doub &prob)
{
	Int j,n=data.size();
	Doub dt,en,ff,fn,fo=0.0;
	KSdist ks;
	sort(data);
	en=n;
	d=0.0;
	for (j=0;j<n;j++) {
		fn=(j+1)/en;
		ff=func(data[j]);
		dt=MAX(abs(fo-ff),abs(fn-ff));
		if (dt > d) d=dt;
		fo=fn;
	}
	en=sqrt(en);
	prob=ks.qks((en+0.12+0.11/en)*d);
}
void kstwo(VecDoub_IO &data1, VecDoub_IO &data2, Doub &d, Doub &prob)
{
	Int j1=0,j2=0,n1=data1.size(),n2=data2.size();
	Doub d1,d2,dt,en1,en2,en,fn1=0.0,fn2=0.0;
	KSdist ks;
	sort(data1);
	sort(data2);
	en1=n1;
	en2=n2;
	d=0.0;
	while (j1 < n1 && j2 < n2) {
		if ((d1=data1[j1]) <= (d2=data2[j2]))
			do
				fn1=++j1/en1;
			while (j1 < n1 && d1 == data1[j1]);
		if (d2 <= d1)
			do
				fn2=++j2/en2;
			while (j2 < n2 && d2 == data2[j2]);
		if ((dt=abs(fn2-fn1)) > d) d=dt;
	}
	en=sqrt(en1*en2/(en1+en2));
	prob=ks.qks((en+0.12+0.11/en)*d);
}
