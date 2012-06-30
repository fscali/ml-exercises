function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


mu=(mean(X))'; %mu must be a column vector but features are in columns, so I have to transpose it
%size(mu)  2 x 1
mu_rep=(repmat(mu,1,m))'; %replicate mu to have the same size of X (now it is m x n because I transpose the replicated matrix)

X_sub=X-mu_rep; %subtract means

sigma2=((1/m)*sum(X_sub.^2))';






% =============================================================


end
