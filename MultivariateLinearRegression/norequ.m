% normal equations
x = load('ex3x.dat');
y = load('ex3y.dat');

x = [ones(size(x,1),1), x]

theta_norequ = inv((x'*x))*x'*y

price_norequ = theta_norequ'*[1 1650 3]'