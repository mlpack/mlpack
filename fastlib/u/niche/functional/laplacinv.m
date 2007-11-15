% laplacinv() - sample from a laplacian distribution

function x = laplacinv(p, mu, b);

x = mu - b * sign(p - 0.5) .* log(1 - 2 * abs(p - 0.5));
