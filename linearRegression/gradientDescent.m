%exercise 2 linearRegression
%gradient descent update 
% x refers to a boy's age
% y is a boy's height in meters

clear all; close all,clc
x = load('ex2x.dat'); y = load('ex2y.dat');
m = length(y);

% plot the traning
figure;
plot(x,y,'o')
ylabel('Height in meters')
xlabel('Age in years')

% gradient descent 
x = [ones(m,1), x];
theta = zeros(size(x(1,:)))';
MAX_ITR = 1500;
alpha = 0.07;

for num_iterations = 1:MAX_ITR
    grad = (1/m).*x'*(x*theta - y);
    theta = theta - alpha.*grad;    
end
%print theta to screeen
theta

%plot the linear fit
hold on;
plot(x(:,2), x*theta, '-')
legend('Traning data', 'Linear regression')
hold off

exact_theta = (x'*x)\x'*y
%predict values for age 3.5 and 7
predict1 = [1, 3.5]*theta
predict2 = [1, 7]*theta

%calculate J matrx
theta0_vals = linspace(-3,3,100);
theta1_vals = linspace(-1,1,100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
    t = [theta0_vals(i); theta1_vals(j)];
    J_vals(i,j) = (0.5/m).*(x*t - y)'*(x*t - y);
    end
        
end

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2,2,15))
xlabel('\theta_0');
ylabel('\theta_1');
