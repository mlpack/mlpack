N=1000;
[roll, roll_color]=swiss_roll(N);
maniplot3(roll, roll_color);
csvwrite('swiss_roll.csv', roll);

unfolded=csvread('results.csv');
maniplot2(unfolded, roll_color);