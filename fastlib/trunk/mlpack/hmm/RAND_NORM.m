u = 2*rand(1, 2000)-1;
v = 2*rand(1, 2000)-1;
r = u.^2+v.^2;
pos = find(r <= 1);
u = u(pos);
v = v(pos);
r = r(pos);
v = sqrt(-2*log(r)./r).*u;