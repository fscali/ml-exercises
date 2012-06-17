function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain). 
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
%   trained SVM model (svmTrain). X is a mxn matrix where there each 
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%



%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

%Cvec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%sigmavec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%error_vec = zeros(size(Cvec,1), size(sigmavec,1));
%error_best = 1000000;
%bestC = Cvec(1);
%bestSigma=sigmavec(1);
%size(Cvec)
%size(sigmavec)
%for i=1:size(Cvec),
%	for j=1:size(sigmavec),
%		disp(sprintf('now trying C= %0.2f, Sigma=%0.2f',Cvec(i), sigmavec(j))); 
%		model = svmTrain(X,y,Cvec(i), @(x1,x2) gaussianKernel(x1,x2,sigmavec(j)));
%		pred = svmPredict(model,Xval);
%		error_vec(i,j) = mean(double(pred ~= yval));
%		if (error_best > error_vec(i,j)),
%			error_best = error_vec(i,j);
%			bestC = Cvec(i);
%			bestSigma = sigmavec(j);
%		end
%	end
%	
%end
%bestC
%bestSigma
C = 1;
sigma=0.1;
% =========================================================================

end
