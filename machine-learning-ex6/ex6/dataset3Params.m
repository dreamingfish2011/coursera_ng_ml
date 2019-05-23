function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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
if(0),
  v = [0.01 0.03 0.1 0.3 1 3 10 30];
  min = inf;
  fprintf('find C,sigma make e minest on val [0.01 0.03 0.1 0.3 1 3 10 30]');
  for c = v,
    for si = v,
      %fprintf('c, s : %f %f\n', c, si);
      model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, si));
      e = mean(double(svmPredict(model, Xval) ~= yval));
      if e <= min,
        min =e;
        C = c;
        sigma = si;
        fprintf('min c, s, e: %f %f %f\n', c, si, e);
      endif

    endfor
  endfor 
  fprintf('final C, sigma, min_error: %f %f %f\n', C, sigma, min);
endif

% =========================================================================

end
