t;


big_spline = spline(t, e1(t));
myshpline = spline(y, e1(y));
intmyshpline = fnint(myshpline);


t1 = clock;

for i=1:1000
  myshpline = spline(y, e1(y));
  ppval(fnint(myshpline), [t(1) t(end)]);
end;

t2 = clock;

t_fnint = t2 - t1


t1 = clock;

for i=1:1000
  big_spline = spline(t, e1(t));
  ppval(fnint(big_spline), [t(1) t(end)]);
end;

t2 = clock;


t_fnintbig = t2 - t1



t1 = clock;
for i=1:1000
  a = spline(t, data(:,1) .* f1);
  ppval(fnint(a), [t(1) t(end)]);
end
t2 = clock;

t_fnintall = t2 - t1






t1 = clock;

for i=1:1000
  quad(@ppval, t(1), t(end), [], [], spline(y, e1(y)));
end

t2 = clock;

t_quad = t2 - t1