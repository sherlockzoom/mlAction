% Exercise: Logistic Regression and Newton's Method 
clear all; close all; clc

x = load('ex4x.dat');
y = load('ex4y.dat');

% x = [ones(size(x,1),1), x]
[m,n] = size(x);
x = [ones(m,1), x];

% find returns the indices of the rows meeting the specified condition
figure
pos = find(y == 1); neg = find( y == 0);
% Assum the features are in the 2nd and 3rd columns of x
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg,3), 'o'); 

% define sigmod function
g = inline('1.0 ./  (1.0 + exp(-z))');

% Initialize fitting parameters
theta = zeros(n+1, 1);

itera_num = 15;
sample_num = size(x,1);
Jtheta = zeros(itera_num, 1);

for i = 1:itera_num
    z = x * theta;
    h = g(z);
    Jtheta(i) = (1/m).*sum(-y.*log(h) - (1 - y).*log(1 - h));
    grad = (1/sample_num).*x'*(h - y);
%     H = (1/m).*x'*h*(1-h)*x; 
    H = (1/m).*x'*diag(h)*diag(1-h)*x;
    theta = theta - H\grad;
end

% 1 - g(x*theta)
prob_test = 1 - g([1, 20, 80]*theta);

% display prob_test theta Jtheta
theta

prob_test

% dcision boundary theta*x=0  ==>> x2 = (-1/w2)*(w1*x1+w0*x0)
plot_x = [min(x(:,2)) - 2, max(x(:,2)) + 2];
plot_y = (-1/theta(3))*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% plot Newton's method 
figure
% plot(Jtheta,'DisplayName','Jtheta','YDataSource','Jtheta');
plot(0:itera_num-1, Jtheta, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
% Display J
Jtheta



