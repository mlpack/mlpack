function c = obj(W, X);

Y = W * X;

c = vasicek_sum(Y) + log(abs(det(W)));

c
W
