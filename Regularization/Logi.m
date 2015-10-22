% Regularized logistic regression
clc,clear

x = load('ex5Logx.dat');
y = load('ex5Logy.dat');

% define sigmod function
g = inline('1.0 ./  (1.0 + exp(-z))');
% Initialize fitting paremeters
x = map_feature(x(:,1), x(:,2));
[m, n] = size(x);
lambda = [0 1 10];
theta = zeros(n, 1);
MAX_ITR = 15;
J = zeros(MAX_ITR, 1);
% Newton's method
for lambda_i = 1:length(lambda)
    x_plot = load('ex5Logx.dat');
    y_plot = load('ex5Logy.dat');
    figure
    % Find the indices for the 2 classes
    pos = find(y_plot); neg = find(y_plot == 0);

    plot(x_plot(pos, 1), x_plot(pos, 2), '+')
    hold on
    plot(x_plot(neg, 1), x_plot(neg, 2), 'o','MarkerFaceColor', 'y')


    for i = 1:MAX_ITR
        z = x*theta;
        h = g(z);
        J(i) = (1/m).*sum(-y.*log(h) - (1 - y).*log(1 - h)) + (lambda(lambda_i)/(2*m))*norm(theta([2:end]))^2;
        % Calculate gradient and hessian
        G = (lambda(lambda_i)/m).*theta; G(1) = 0;
        L = (lambda(lambda_i)/m)*eye(n); L(1) = 0;
        grad = ((1/m).*x'*(h - y)) + G;
        H = ((1/m).*x'*diag(h)*diag(1 - h)*x) + L;

        theta = theta - H\grad;
    end
    % Plot
    J;
    norm_theta = norm(theta);
    % Define the range of the grid
    u = linspace(-1, 1.5, 200);
    v = linspace(-1, 1.5, 200);
    %Initialize space for the values to be plotted
    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for k = 1:length(u)
        for j = 1:length(v)
            z(k,j) = map_feature(u(k),v(j))*theta;
        end
    end
    z = z';
    contour(u,v,z, [0, 0], 'LineWidth',2)
    legend('y=1', 'y=0', 'Decision boundary');
    title(sprintf('\\lambda = %g', lambda(lambda_i)), 'FontSize', 14)
    hold off

end
