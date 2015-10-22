%Multivariate Linear Regression
x = load('ex3x.dat');
y = load('ex3y.dat');

% add the x0=1 intercept term into your x matrix
x = [ones(size(x,1),1), x];

sigma = std(x); %standard deviations
mu = mean(x); % means

x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

% gradient descent
figure
alpha = [0.01,0.03,0.1,0.3,1,1.3]
J =zeros(50, 1);
itera_num = 100; 
sample_num = size(x,1);
plotstyle = {'b','r','g','k','b--', 'r--'};

theta_grad_descent = zeros(size(x(1,:)));

for alpha_i = 1:length(alpha)
    theta = zeros(size(x(1,:)))';
    Jtheta = zeros(itera_num,1);
    for i = 1:itera_num
        Jtheta(i) = (0.5/sample_num).*(x*theta - y)'*(x*theta - y);
        grad = (1/sample_num).*x'*(x*theta - y);
        theta = theta - alpha(alpha_i).*grad;
    end
    plot(0:49, Jtheta(1:50),char(plotstyle(alpha_i)), 'LineWidth', 2)
    hold on
    if (1==alpha(alpha_i))
        theta_grad_descent = theta
    end
end

legend('0.01', '0.03', '0.1', '0.3', '1', '1.3');

% now plot J
% technically, the first J starts at the zero-eth iteration

xlabel('Number of iterations')
ylabel('Cost J')

% predict price
price_grad_descent = theta_grad_descent'*[1 (1650-mu(2))/sigma(2) (3-mu(3))/sigma(3)]'
