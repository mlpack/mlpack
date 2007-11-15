% laplacpdf() - laplacian probability density function

function p = laplacpdf(x, mu, b);

p = (1 / (2 * b)) * exp(-1 * abs(x - mu) / b);
