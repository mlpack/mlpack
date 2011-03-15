void moment(VecDoub_I &data, Doub &ave, Doub &adev, Doub &sdev, Doub &var,
	Doub &skew, Doub &curt) {
	Int j,n=data.size();
	Doub ep=0.0,s,p;
	if (n <= 1) throw("n must be at least 2 in moment");
	s=0.0;
	for (j=0;j<n;j++) s += data[j];
	ave=s/n;
	adev=var=skew=curt=0.0;
	for (j=0;j<n;j++) {
		adev += abs(s=data[j]-ave);
		ep += s;
		var += (p=s*s);
		skew += (p *= s);
		curt += (p *= s);
	}
	adev /= n;
	var=(var-ep*ep/n)/(n-1);
	sdev=sqrt(var);
	if (var != 0.0) {
		skew /= (n*var*sdev);
		curt=curt/(n*var*var)-3.0;
	} else throw("No skew/kurtosis when variance = 0 (in moment)");
}
void avevar(VecDoub_I &data, Doub &ave, Doub &var) {
	Doub s,ep;
	Int j,n=data.size();
	ave=0.0;
	for (j=0;j<n;j++) ave += data[j];
	ave /= n;
	var=ep=0.0;
	for (j=0;j<n;j++) {
		s=data[j]-ave;
		ep += s;
		var += s*s;
	}
	var=(var-ep*ep/n)/(n-1);
}
