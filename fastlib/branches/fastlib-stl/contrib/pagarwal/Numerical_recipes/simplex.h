extern "C" {
	#include "lusol.h"
}

struct NRlusol
{
	LUSOLrec *LUSOL;
	Int inform;

	NRlusol(Int m, Int nz);
	void load_col(const Int col, VecInt &row_ind, VecDoub &val);
	void factorize();
	VecDoub solve(VecDoub &rhs);
	VecDoub solvet(VecDoub &rhs);
	VecDoub linv(VecDoub &rhs);
	VecDoub uinv(VecDoub &rhs);
	VecDoub linvt(VecDoub &rhs);
	VecDoub uinvt(VecDoub &rhs);
	void update(VecDoub &x, Int i, Int &ok);
	void clear();
	~NRlusol();
};

NRlusol::NRlusol(Int m, Int nz) {
	LUSOL = LUSOL_create(stdout, 0, LUSOL_PIVMOD_TPP, 0);
	LUSOL->luparm[LUSOL_IP_SCALAR_NZA] = 10;
	LUSOL->parmlu[LUSOL_RP_FACTORMAX_Lij] = 5.0;
	LUSOL->parmlu[LUSOL_RP_UPDATEMAX_Lij] = 5.0;
	LUSOL_sizeto(LUSOL, m, m, nz);
	LUSOL->m = m;
	LUSOL->n = m;
	LUSOL->nelem = nz;
}

void NRlusol::load_col(Int col, VecInt &row_ind, VecDoub &val)
{
	Int nz=row_ind.size()-1;
	Int status=LUSOL_loadColumn(LUSOL,&row_ind[0],col,&val[0],nz,0);
}

void NRlusol::factorize()
{
	LU1FAC(LUSOL, &inform );
	if (inform > LUSOL_INFORM_SERIOUS) {
		cout << "    Error:" << endl << LUSOL_informstr(LUSOL, inform) << endl;
		throw("LUSOL exiting");
	}
}

VecDoub NRlusol::solve(VecDoub &rhs)
{
	VecDoub x(rhs.size()),y=rhs;
	LU6SOL(LUSOL,LUSOL_SOLVE_Aw_v,&y[0],&x[0], NULL, &inform);
	return x;
}

VecDoub NRlusol::solvet(VecDoub &rhs)
{
	VecDoub x(rhs.size()),y=rhs;
	LU6SOL(LUSOL,LUSOL_SOLVE_Atv_w,&x[0],&y[0], NULL, &inform);
	return x;
}

VecDoub NRlusol::linv(VecDoub &rhs)
{
	VecDoub x=rhs;
	LU6SOL(LUSOL,LUSOL_SOLVE_Lv_v,&x[0],&x[0], NULL, &inform);
	return x;
}

VecDoub NRlusol::uinv(VecDoub &rhs)
{
	VecDoub x(rhs.size());
	LU6SOL(LUSOL,LUSOL_SOLVE_Uw_v,&rhs[0],&x[0], NULL, &inform);
	return x;
}

VecDoub NRlusol::linvt(VecDoub &rhs)
{
	VecDoub x=rhs;
	LU6SOL(LUSOL,LUSOL_SOLVE_Ltv_v,&x[0],&x[0], NULL, &inform);
	return x;
}

VecDoub NRlusol::uinvt(VecDoub &rhs)
{
	VecDoub x(rhs.size()),y=rhs;
	LU6SOL(LUSOL,LUSOL_SOLVE_Utv_w,&x[0],&y[0], NULL, &inform);
	return x;
}

void NRlusol::update(VecDoub &x, Int i, Int &ok)
{
	Doub DIAG, VNORM;
	LU8RPC(LUSOL, LUSOL_UPDATE_OLDNONEMPTY, LUSOL_UPDATE_USEPREPARED,
		i, &x[0], NULL, &ok, &DIAG, &VNORM);
}

void NRlusol::clear()
{
	LUSOL_clear(LUSOL, TRUE);
}

NRlusol::~NRlusol()
{
	LUSOL_free(LUSOL);
}
struct Simplex {
	Int m,n,ierr;
	VecInt initvars,ord,ad;
	VecDoub b,obj,u,x,xb,v,w,scale;
	NRsparseMat &sa;
	Int NMAXFAC,NREFAC;
	Doub EPS,EPSSMALL,EPSARTIF,EPSFEAS,EPSOPT,EPSINFEAS,EPSROW1,EPSROW2,EPSROW3;
	Int nm1,nmax,nsteps;
	Bool verbose;
	NRvector<NRsparseCol *> a;
	NRlusol *lu;

	Simplex(Int mm,Int nn,VecInt_I &initv,VecDoub_I &bb,VecDoub_I &objj,
		NRsparseMat &ssa,Bool verb);
	void solve();
	void initialize();
	void scaleit();
	void phase0();
	void phase1();
	void phase2();
	Int colpiv(VecDoub &v,Int phase,Doub &piv);
	Int rowpiv(VecDoub_I &w,Int phase,Int kp,Doub xbmax);
	void transform(VecDoub &x,Int ip,Int kp);
	Doub maxnorm(VecDoub_I &xb);
	VecDoub getcol(Int k);
	VecDoub lx(Int kp);
	Doub xdotcol(VecDoub_I &x, Int k);
	void refactorize();
	void prepare_output();
};

Simplex::Simplex(Int mm,Int nn,VecInt_I &initv,VecDoub_I &bb,VecDoub_I &objj,
		NRsparseMat &ssa,Bool verb) :
	m(mm),n(nn),initvars(initv),b(bb),obj(objj),sa(ssa),verbose(verb),ord(m+1),
	ad(m+n+1),u(m+n+1),x(m+1),xb(m+1),v(m+1),w(m+1),scale(n+m+1),a(n+1) {
	NMAXFAC=40;
	NREFAC=50;
	EPS=numeric_limits<Doub>::epsilon();
	EPSSMALL=1.0e5*EPS;
	EPSARTIF=1.0e5*EPS;
	EPSFEAS=1.0e8*EPS;
	EPSOPT=1.0e8*EPS;
	EPSINFEAS=1.0e4*EPS;
	EPSROW1=1.0e-5;
	EPSROW2=EPS;
	EPSROW3=EPS;
}

void Simplex::solve()
{
	initialize();
	scaleit();
	phase0();
	if (verbose)
		cout << "    at end of phase0,iter= " << nsteps << endl;
	if (ierr != 0) {
		return;
	}
	phase1();
	if (verbose)
		cout << "    at end of phase1,iter= " << nsteps << endl;
	if (ierr != 0) {
		return;
	}
	phase2();
	prepare_output();
}

void Simplex::initialize() {
	VecInt irow(2);
	VecDoub value(2);
	nsteps=0;
	ierr=0;
	nm1=n+m+1;
	nmax=NMAXFAC*MAX(m,n);
	for (Int i=1;i<=n;i++)
		if (initvars[i] == 0)
			ad[i]=0;
		else
			ad[i]=1;
	for (Int i=n+1;i<=n+m;i++)
		ad[i]=-1;
	for (Int i=1;i<=m;i++)
		if (initvars[n+i] >= 0)
			ord[i]=n+i;
		else
			ord[i]=-m+i-1;
	for (Int i=0;i<n;i++) {
		Int nvals=sa.col_ptr[i+1]-sa.col_ptr[i];
		a[i+1]=new NRsparseCol(m+1,nvals+1);
		Int count=1;
		for (Int j=sa.col_ptr[i]; j<sa.col_ptr[i+1]; j++) {
			Int k=sa.row_ind[j];
			a[i+1]->row_ind[count]=k+1;
			a[i+1]->val[count]=sa.val[j];
			count++;
		}
	}
	lu=new NRlusol(m,sa.col_ptr[n]);
	value[0]=0.0;
	value[1]=1.0;
	irow[0]=0;
	for (Int i=1;i<=m;i++) {
		irow[1]=i;
		lu->load_col(i,irow,value);
	}
	lu->factorize();
}

void Simplex::scaleit()
{
	for (Int i=1;i<=m;i++)
		scale[n+i]=0.0;
	for (Int k=1;k<=n;k++) {
		x=getcol(k);
		Doub h=0.0;
		for (Int i=1;i<=m;i++)
			if (abs(x[i]) > h)
				h=abs(x[i]);
		if (h == 0.0)
			scale[k]=0.0;
		else
			scale[k]=1.0/h;
		for (Int i=1;i<=m;i++)
			scale[n+i]=MAX(scale[n+i],abs(x[i])*scale[k]);
	}
	for (Int i=1;i<=m;i++)
		if (scale[n+i] == 0.0)
			scale[n+i]=1.0;
}

void Simplex::phase0()
{
	Int ind,ip,kp;
	Doub piv;
	for (kp=1;kp<=n;kp++) {
		if (initvars[kp] < 0) {
			x=lx(kp);
			w=lu->uinv(x);
			ip=rowpiv(w,0,kp,0.0);
			ind=ord[ip];
			transform(x,ip,kp);
			ord[ip] -= nm1;
			if (initvars[ind] == 0)
				ad[ind]=0;
		}
	}
	for (ip=1;ip<=m;ip++) {
		ind=ord[ip];
		if (ind < 0 || initvars[ind] != 0)
			continue;
		for (Int i=1;i<=m;i++) v[i]=0.0;
		v[ip]=1.0;
		kp=colpiv(v,0,piv);
		if (abs(piv) < EPSSMALL) {
			xb=lu->solve(b);
			if (abs(xb[ip]) > EPSARTIF*maxnorm(xb)) {
				ierr=1;
				return;
			} else {
				if (verbose)
					cout << "    artificial variable remains: ip " << ip << endl;
				continue;
			}
		}
		x=lx(kp);
		transform(x,ip,kp);
		if (ad[ind] == 1)
			ad[ind]=0;
	}
}

void Simplex::phase1()
{
	Int ip,kp;
	Doub piv;
	for (;;) {
		xb=lu->solve(b);
		Doub xbmax=maxnorm(xb);
		Bool done=true;
		for (Int i=1;i<=m;i++) {
			if (ord[i] > 0 && xb[i] < -EPSFEAS*xbmax) {
				v[i]=1.0/scale[ord[i]];
				done=false;
			}
  			else
				v[i]=0.0;
		}
		if (done)
			break;
		kp=colpiv(v,1,piv);
		if (ierr != 0)
			return;
		Bool first=true;
		for (;;) {
			x=lx(kp);
			w=lu->uinv(x);
			ip=rowpiv(w,1,kp,xbmax);
			if (ierr == 0)
				break;
			if (!first) {
				ierr=6;
				return;
			}
			if (verbose)
				cout << "    attempt to recover" << endl;
			ierr=0;
			first=false;
			refactorize();
		}
		transform(x,ip,kp);
		if (nsteps >= nmax) {
			ierr=3;
			return;
		}
	}
}

void Simplex::phase2()
{
	Int ip,kp;
	Doub piv;
	for (;;) {
		for (Int i=1;i<=m;i++) {
			if (ord[i] > 0)
				v[i]=-obj[ord[i]];
  			else
				v[i]=-obj[ord[i]+nm1];
		}
		kp=colpiv(v,2,piv);
		if (piv > -EPSOPT)
			break;
		Bool first=true;
		for (;;) {
			x=lx(kp);
			w=lu->uinv(x);
			xb=lu->solve(b);
			Doub xbmax=maxnorm(xb);
			ip=rowpiv(w,2,kp,xbmax);
			if (ierr == 0)
				break;
			if (!first)
				return;
			if (verbose)
				cout << "    attempt to recover" << endl;
			ierr=0;
			first=false;
			refactorize();
		}
		transform(x,ip,kp);
		if (verbose) {
			prepare_output();
			cout << "    in phase2,iter,obj. fn. " << nsteps << " " << u[0] << endl;
		}
		if (nsteps >= nmax) {
			prepare_output();
			cout << "    in phase2,iter,obj. fn. " << nsteps << " " << u[0] << endl;
			ierr=4;
			return;
		}
	}
}

Int Simplex::colpiv(VecDoub &v,Int phase,Doub &piv)
{
	Int kp;
	Doub h1;
	x=lu->solvet(v);
	piv=0.0;
	for (Int k=1;k<=n+m;k++) {
		if (ad[k] > 0) {
			if (k > n)
				h1=x[k-n];
			else
				h1=xdotcol(x,k);
			if (phase == 2)
				h1=h1+obj[k];
			h1=h1*scale[k];
			if ((phase == 0 && abs(h1) > abs(piv)) ||
					(phase > 0 && h1 < piv)) {
				piv=h1;
				kp=k;
			}
		}
	}
	if (phase == 1) {
		h1=0.0;
		for (Int k=1;k<=m;k++)
			h1 += abs(x[k])*scale[n+k];
		if (piv > -EPSINFEAS*h1)
			ierr=2;
	}
	return kp;
}

Int Simplex::rowpiv(VecDoub_I &w,Int phase,Int kp,Doub xbmax)
{
	Int j=0,ip=0;
	Doub h1,h2,min=0.0,piv=0.0;
	for (Int i=1;i<=m;i++) {
		Int ind=ord[i];
		if (ind > 0) {
			if (abs(w[i])*scale[kp] <= EPSROW1*scale[ind])
				continue;
			if (phase == 0) {
				h1=abs(w[i]);
				h2=h1;
			} else {
				Doub hmin=EPSROW2*xbmax*scale[ind]*m;
				if (abs(xb[i]) < hmin)
					h2=hmin;
				else
					h2=xb[i];
				h1=w[i]/h2;
			}
			if (h1 > 0.0) {
				if (h2 > 0.0 && h1 > piv) {
					piv=h1;
					ip=i;
				} else if ((h2*scale[kp] < -EPSROW3*scale[ind]) &&
					(j == 0 || h1 < min)) {
						min=h1;
						j=i;
				}
			}
		}
	}
	if (min > piv) {
		piv=min;
		ip=j;
	}
	if (ip == 0)
		ierr=5;
	return ip;
}

void Simplex::transform(VecDoub &x,Int ip,Int kp)
{
	ad[ord[ip]]=1;
	ad[kp]=-1;
	Int oldord=ord[ip];
	ord[ip]=kp;
	nsteps++;
	if ((nsteps % NREFAC) != 0) {
		Int ok;
		lu->update(x,ip,ok);
		if (ok != 0) {
			if (verbose)
				cout << "    singular update, refactorize" << endl;
			ad[oldord]=-1;
			ad[kp]=1;
			ord[ip]=oldord;
			refactorize();
		}
	}
	else
		refactorize();
}

Doub Simplex::maxnorm(VecDoub_I &xb)
{
	Int indv;
	Doub maxn=0.0;
	for (Int i=1;i<=m;i++) {
		if (ord[i] > 0)
			indv=ord[i];
		else
			indv=ord[i]+nm1;
		Doub test=abs(xb[i])/scale[indv];
		if (test > maxn)
			maxn=test;
	}
	if (maxn != 0.0)
		return maxn;
	else
		return 1.0;
}

VecDoub Simplex::getcol(Int k)
{
	VecDoub temp(m+1,0.0);
	for (Int i=1;i<a[k]->nvals;i++) {
		temp[a[k]->row_ind[i]]=a[k]->val[i];
	}
	return temp;
}

VecDoub Simplex::lx(Int kp)
{
	if (kp <= n)
		x=getcol(kp);
	else {
		for (Int i=1;i<=m;i++) x[i]=0.0;
		x[kp-n]=1.0;
	}
	return lu->linv(x);
}

Doub Simplex::xdotcol(VecDoub_I &x,Int k)
{
	Doub sum=0.0;
	for (Int i=1;i<a[k]->nvals;i++)
		sum += x[a[k]->row_ind[i]]*a[k]->val[i];
	return sum;
}

void Simplex::refactorize()
{
	Int count=0;
	Doub sum=0.0;
	VecInt irow(2);
	VecDoub value(2);
	value[0]=0.0;
	value[1]=1.0;
	irow[0]=0;
	lu->clear();
	for (Int i=1;i<=m;i++) {
		Int ind=ord[i];
		if (ind < 0)
			ind += nm1;
		if (ind > n) {
			irow[1]=ind-n;
			lu->load_col(i,irow,value);
			count++;
			sum += 1.0;
		}
		else {
			lu->load_col(i,a[ind]->row_ind,a[ind]->val);
			for (Int j=1;j<a[ind]->nvals;j++) {
				count++;
				sum += abs(a[ind]->val[j]);
			}
		}
	}
	Doub small=EPSSMALL*sum/count;
	lu->LUSOL->parmlu[LUSOL_RP_SMALLDIAG_U]    =
		lu->LUSOL->parmlu[LUSOL_RP_EPSDIAG_U]  = small;
	lu->factorize();
}

void Simplex::prepare_output()
{
	Int indv;
	Doub sum=obj[0];
	xb=lu->solve(b);
	for (Int i=0;i<=m+n;i++) u[i]=0.0;
	for (Int i=1;i<=m;i++) {
		if (ord[i] > 0)
			indv=ord[i];
		else
			indv=ord[i]+nm1;
		u[indv]=xb[i];
		sum += xb[i]*obj[indv];
	}
	u[0]=sum;
}
