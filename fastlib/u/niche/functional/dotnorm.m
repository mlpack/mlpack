% dotnorm() - normalize vectors and take dot product

function w = dotnorm(u, v);

w = dot(u/norm(u), v/norm(v));