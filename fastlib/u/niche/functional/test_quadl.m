t = linspace(0,1,30);

x = sin(t);

y = cos(t);


spline1 = spline(t, sin(t));
spline2 = spline(t, cos(t));

t1 = clock;
for i=1:1000
  quadl(@myfunk, 0, 1, [], [], spline1, spline2);
end
t2 = clock;
t2 - t1