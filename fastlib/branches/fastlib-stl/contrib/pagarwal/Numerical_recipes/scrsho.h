template<class T>
void scrsho(T &fx) {
	const Int RES=500;
	const Doub XLL=75., XUR=525., YLL=250., YUR=700.;
	char *plotfilename = tmpnam(NULL);
	VecDoub xx(RES), yy(RES);
	Doub x1,x2;
	Int i;
	for (;;) {
		Doub ymax = -9.99e99, ymin = 9.99e99, del;
		cout << endl << "Enter x1 x2 (x1=x2 to stop):" << endl;
		cin >> x1 >> x2;
		if (x1==x2) break;
		for (i=0;i<RES;i++) {
			xx[i] = x1 + i*(x2-x1)/(RES-1.);
			yy[i] = fx(xx[i]);
			if (yy[i] > ymax) ymax=yy[i];
			if (yy[i] < ymin) ymin=yy[i];
		}
		del = 0.05*((ymax-ymin)+(ymax==ymin ? abs(ymax) : 0.));
		PSpage pg(plotfilename);
		PSplot plot(pg,XLL,XUR,YLL,YUR);
		plot.setlimits(x1,x2,ymin-del,ymax+del);
		plot.frame();
		plot.autoscales();
		plot.lineplot(xx,yy);
		if (ymax*ymin < 0.) plot.lineseg(x1,0.,x2,0.);
		plot.display();
	}
	remove(plotfilename);
}
