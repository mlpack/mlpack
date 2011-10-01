function z = ComputePoissonZ(D, s, t, Lambda)
%function z = ComputePoissonZ(D, s, t, Lambda)

d = size(D, 1);

z = (t ./ Lambda) - ones(d, 1) + D * s;
