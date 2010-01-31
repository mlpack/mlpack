function [roll, roll_color]=swiss_roll(N)
t=rand(N,1)*3*pi+3*pi/2;
h=rand(N,1)*21;
roll=[t.*cos(t), h, t.*sin(t)];
roll_color=t/max(t);