struct Hashfn1 {
	Ranhash hasher;
	Int n;
	Hashfn1(Int nn) : n(nn) {}
	Ullong fn(const void *key) {
		Uint *k;
		Ullong *kk;
		switch (n) {
			case 4:
				k = (Uint *)key;
				return hasher.int64(*k);
			case 8:
				kk = (Ullong *)key;\
				return hasher.int64(*kk);
			default:
				throw("Hashfn1 is for 4 or 8 byte keys only.");
		}
	}
};
struct Hashfn2 {
	static Ullong hashfn_tab[256];
	Ullong h;
	Int n;
	Hashfn2(Int nn) : n(nn) {
		if (n == 1) n = 0;
		h = 0x544B2FBACAAF1684LL;
		for (Int j=0; j<256; j++) {
			for (Int i=0; i<31; i++) {
				h = (h >>  7) ^ h;
				h = (h << 11) ^ h;
				h = (h >> 10) ^ h;
			}
			hashfn_tab[j] = h;
		}
	}
	Ullong fn(const void *key) {
		Int j;
		char *k = (char *)key;
		h=0xBB40E64DA205B064LL;
		j=0;
		while (n ? j++ < n : *k) {
			h = (h * 7664345821815920749LL) ^ hashfn_tab[(unsigned char)(*k)];
			k++;
		}
		return h;
	}
};
Ullong Hashfn2::hashfn_tab[256];
template<class keyT, class hfnT> struct Hashtable {
	Int nhash, nmax, nn, ng;
	VecInt htable, next, garbg;
	VecUllong thehash;
	hfnT hash;
	Hashtable(Int nh, Int nv);
	Int iget(const keyT &key);
	Int iset(const keyT &key);
	Int ierase(const keyT &key);
	Int ireserve();
	Int irelinquish(Int k);
};

template<class keyT, class hfnT>
Hashtable<keyT,hfnT>::Hashtable(Int nh, Int nv):
	hash(sizeof(keyT)), nhash(nh), nmax(nv), nn(0), ng(0),
	htable(nh), next(nv), garbg(nv), thehash(nv) {
	for (Int j=0; j<nh; j++) { htable[j] = -1; }
}
template<class keyT, class hfnT>
Int Hashtable<keyT,hfnT>::iget(const keyT &key) {
	Int j,k;
	Ullong pp = hash.fn(&key);
	j = (Int)(pp % nhash);
	for (k = htable[j]; k != -1; k = next[k]) {
		if (thehash[k] == pp) {
			return k;
		}
	}
	return -1;
}
template<class keyT, class hfnT>
Int Hashtable<keyT,hfnT>::iset(const keyT &key) {
	Int j,k,kprev;
	Ullong pp = hash.fn(&key);
	j = (Int)(pp % nhash);
	if (htable[j] == -1) {
		k = ng ? garbg[--ng] : nn++ ;
		htable[j] = k;
	} else {
		for (k = htable[j]; k != -1; k = next[k]) {
			if (thehash[k] == pp) {
				return k;
			}
			kprev = k;
		}
		k = ng ? garbg[--ng] : nn++ ;
		next[kprev] = k;
	}
	if (k >= nmax) throw("storing too many values");
	thehash[k] = pp;
	next[k] = -1;
	return k;
}
template<class keyT, class hfnT>
Int Hashtable<keyT,hfnT>::ierase(const keyT &key) {
	Int j,k,kprev;
	Ullong pp = hash.fn(&key);
	j = (Int)(pp % nhash);
	if (htable[j] == -1) return -1;
	kprev = -1;
	for (k = htable[j]; k != -1; k = next[k]) {
		if (thehash[k] == pp) {
			if (kprev == -1) htable[j] = next[k];
			else next[kprev] = next[k];
			garbg[ng++] = k;
			return k;
		}
		kprev = k;
	}
	return -1;
}
template<class keyT, class hfnT>
Int Hashtable<keyT,hfnT>::ireserve() {
	Int k = ng ? garbg[--ng] : nn++ ;
	if (k >= nmax) throw("reserving too many values");
	next[k] = -2;
	return k;
}

template<class keyT, class hfnT>
Int Hashtable<keyT,hfnT>::irelinquish(Int k) {
	if (next[k] != -2) {return -1;}
	garbg[ng++] = k;
	return k;
}
template<class keyT, class elT, class hfnT>
struct Hash : Hashtable<keyT, hfnT> {
	using Hashtable<keyT,hfnT>::iget;
	using Hashtable<keyT,hfnT>::iset;
	using Hashtable<keyT,hfnT>::ierase;
	vector<elT> els;

	Hash(Int nh, Int nm) : Hashtable<keyT, hfnT>(nh, nm), els(nm) {}

	void set(const keyT &key, const elT &el)
		{els[iset(key)] = el;}

	Int get(const keyT &key, elT &el) {
		Int ll = iget(key);
		if (ll < 0) return 0;
		el = els[ll];
		return 1;
	}

	elT& operator[] (const keyT &key) {
		Int ll = iget(key);
		if (ll < 0) {
			ll = iset(key);
			els[ll] = elT();
		}
		return els[ll];
	}

	Int count(const keyT &key) {
		Int ll = iget(key);
		return (ll < 0 ? 0 : 1);
	}

	Int erase(const keyT &key) {
		return (ierase(key) < 0 ? 0 : 1);
	}
};
template<class keyT, class elT, class hfnT>
struct Mhash : Hashtable<keyT,hfnT> {
	using Hashtable<keyT,hfnT>::iget;
	using Hashtable<keyT,hfnT>::iset;
	using Hashtable<keyT,hfnT>::ierase;
	using Hashtable<keyT,hfnT>::ireserve;
	using Hashtable<keyT,hfnT>::irelinquish;
	vector<elT> els;
	VecInt nextsis;
	Int nextget;
	Mhash(Int nh, Int nm);
	Int store(const keyT &key, const elT &el);
	Int erase(const keyT &key, const elT &el);
	Int count(const keyT &key);
	Int getinit(const keyT &key);
	Int getnext(elT &el);
};

template<class keyT, class elT, class hfnT>
Mhash<keyT,elT,hfnT>::Mhash(Int nh, Int nm)
	: Hashtable<keyT, hfnT>(nh, nm), nextget(-1), els(nm), nextsis(nm) {	
	for (Int j=0; j<nm; j++) {nextsis[j] = -2;}
}

template<class keyT, class elT, class hfnT>
Int Mhash<keyT,elT,hfnT>::store(const keyT &key, const elT &el) {
	Int j,k;
	j = iset(key);
	if (nextsis[j] == -2) {
		els[j] = el;
		nextsis[j] = -1;
		return j;
	} else {
		while (nextsis[j] != -1) {j = nextsis[j];}
		k = ireserve();
		els[k] = el;
		nextsis[j] = k;
		nextsis[k] = -1;
		return k;
	}
}

template<class keyT, class elT, class hfnT>
Int Mhash<keyT,elT,hfnT>::erase(const keyT &key, const elT &el) {
	Int j = -1,kp = -1,kpp = -1;
	Int k = iget(key);
	while (k >= 0) {
		if (j < 0 && el == els[k]) j = k;
		kpp = kp;
		kp = k;
		k=nextsis[k];
	}
	if (j < 0) return 0;
	if (kpp < 0) {
		ierase(key);
		nextsis[j] = -2;
	} else {
		if (j != kp) els[j] = els[kp];
		nextsis[kpp] = -1;
		irelinquish(kp);
		nextsis[kp] = -2;
	}
	return 1;
}

template<class keyT, class elT, class hfnT>
Int Mhash<keyT,elT,hfnT>::count(const keyT &key) {
	Int next, n = 1;
	if ((next = iget(key)) < 0) return 0;
	while ((next = nextsis[next]) >= 0)  {n++;}
	return n;
}

template<class keyT, class elT, class hfnT>
Int Mhash<keyT,elT,hfnT>::getinit(const keyT &key) {
	nextget = iget(key);
	return ((nextget < 0)? 0 : 1);
}

template<class keyT, class elT, class hfnT>
Int Mhash<keyT,elT,hfnT>::getnext(elT &el) {
	if (nextget < 0) {return 0;}
	el = els[nextget];
	nextget = nextsis[nextget];
	return 1;
}
