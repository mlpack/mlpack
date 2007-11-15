% myfunk() - my function
function x = myfunk(t, splineA, splineB);

x = ppval(splineA, t) .* ppval(splineB, t);