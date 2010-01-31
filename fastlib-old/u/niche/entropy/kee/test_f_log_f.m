N = 1e6;

x = -10;
y = 10;

c_x = 0;
c_y = 0;

Z = rand(N,1);

c_x = sum(Z < .01);
c_y = N - c_x;


p_x = c_x / N;
p_y = c_y / N;

E_x = p_x * x + p_y * y;
E_log_f_x = p_x * log(p_x) * x + p_y * log(p_y) * y;


disp('E[x] = ');
disp(E_x);

disp('E[log(f) x] = ');
disp(E_log_f_x);


