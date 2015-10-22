% Regularized linear regression
clc,clear

x = load('ex5Linx.dat');
y = load('ex5Liny.dat');
% plot the data
plot(x,y,'o','MarkerFaceColor','r')

x = [ones(size(x,1),1), x, x.^2, x.^3, x.^4, x.^5];
[m,n] = size(x);

diag_m = diag([0;ones(n-1,1)]);

lambda = [0 1 10]';

colortype = {'g', 'b', 'r'};

theta =zeros(n,3)
xrange = linspace(min(x(:,2)), max(x(:,2)))';
hold on
% normal equations
for lambda_i = 1:length(lambda)
    theta(:,lambda_i) = inv(x'*x + lambda(lambda_i).*diag_m)*x'*y;
    yrange = [ones(size(xrange)) xrange xrange.^2 xrange.^3 xrange.^4 xrange.^5]*theta(:,lambda_i);
    plot(xrange',yrange,char(colortype(lambda_i)))
    hold on
end
legend('traning data', '\lambda=0', '\lambda=1', '\lambda=10')
hold off