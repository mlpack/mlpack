% gauss_kernel() - gaussian kernel
function p = gauss_kernel(x, h);

p = exp(-(x^2) / (2 * (h^2))) / (h * sqrt(2 * pi));
