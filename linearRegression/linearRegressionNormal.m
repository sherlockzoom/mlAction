x = load('ex2x.dat');
y = load('ex2y.dat');

figure
plot(x,y,'o')
xlabel('Age in years')
ylabel('Height in meters')

m = length(y);  %store the number of training examples
x = [ones(m,1), x]; %Add a colum of ones to x
w = inv(x'*x)*x'*y
hold on
% w = 0.7502 0.0639
plot(x(:,2), 0.0639*x(:,2) + 0.7502)
