function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%Part1

% I have to add an extra columng because the X matrix is an m x n matrix where each row is an example 
% and each column i is the value of the feature i for that example
Xpriv = [ones(m,1) X]; % m X n+1 matrix
%Theta1 is 25 X n+1, so in order to multiply I have to transpose Xpriv'
z2 = Theta1 * Xpriv';
a2 = sigmoid(z2); %25 x m (25 is the size of the input layer)
a2 = [ones(1,m); a2];  %26 x m 

%Theta2 is k X 26, so in order to multiply I don't have to transpose
z3 = Theta2 * a2;
z3 = sigmoid(z3);
h = z3;

%now I can compute the coste 
mySum = 0.0;

%for each example i
for i=1:m,
	% I have to calculate a k sized vector where each value is 0 except the one with the index corresponding to the value of the example i
	Yi=zeros(num_labels,1);
	Yi(y(i,1)) = 1;

	%Yi -> k X 1
	%h  -> k x m
        %h(:,i) -> k x 1
	
	%so I can transpose Yi and multiply for the columng vector of h corresponding to example i
	
	app = - (Yi' * log(h(:,i))) - ((1-Yi)'*log(1-h(:,i)));	
	mySum = mySum + app;
end

J = mySum / m;
% to calculate the regularization term, I split Theta1 and Theta2 and build a temporary vector to store the square products Theta1(j,k) and Theta2(j,k), where j is the number of elements in the destination layer and k is the number of elements in the starting layer
regVec = zeros(size(Theta1,1),1);
for i=1:size(Theta1,1),
	regVec(i,1) = Theta1(i,2:size(Theta1,2)) * (Theta1(i,2:size(Theta1,2)))';
end
regTerm1=sum(regVec);
regVec = zeros(size(Theta2,1),1);
for i=1:size(Theta2,1),
	regVec(i,1) = Theta2(i,2:size(Theta2,2)) * (Theta2(i,2:size(Theta2,2)))';
end
regTerm2=sum(regVec);
regTerm=(lambda/(2*m)) * (regTerm1 + regTerm2);
J = J + regTerm;

% part 2

% K x 26
DELTA2 = zeros(size(Theta2));


% 25 x n+1
DELTA1 = zeros(size(Theta1));

for t = 1:m,
	Yt=zeros(num_labels,1);
        Yt(y(t,1)) = 1;
	%h(:,t) is a3 for the current training example
	delta3=h(:,t) - Yt;
	delta2=((Theta2)' * delta3) .*  [0; sigmoidGradient(z2(:,t))]; 
	DELTA2 = DELTA2 + delta3 * (a2(:,t))';
	DELTA1 = DELTA1 + delta2(2:end) * Xpriv(t,:);

end



Theta1_grad = DELTA1./m;
Theta2_grad = DELTA2./m;

Theta1Reg = [zeros(size(Theta1,1),1)  (lambda/m) * Theta1(:,2:size(Theta1,2))];
Theta2Reg = [zeros(size(Theta2,1),1)  (lambda/m) * Theta2(:,2:size(Theta2,2))];

Theta1_grad = Theta1_grad + Theta1Reg;
Theta2_grad = Theta2_grad + Theta2Reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
