#include <iostream>
#include "Numerical_recipes/nr3.h"
#include "Numerical_recipes/calendar.h"
#include "Numerical_recipes/moment.h"

using namespace std;

Int main(void) {
        const Int NTOT = 20;
        Int i, jd, nph = 2;
        Doub frac, ave, vrnce;
        VecDoub data(NTOT);
        for (i = 0; i < NTOT; i++) {
                flmoon(i, nph, jd, frac);
                data[i] = jd;
        }
        avevar(data, ave, vrnce);
        cout << "Average = " << setw(12) << ave << endl;
        cout << " Variance = " << setw(13) << vrnce << endl;
        return 0;
}