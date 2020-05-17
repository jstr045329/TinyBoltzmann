% TinyBoltzmann.m
% Author: Jim Strieter
% May 16, 2020
% 
% This is a tiny Restricted Boltzmann Machine library with almost no
% dependencies. You need sigmoid.m and normalizeMu0Sigma1.m, both of which
% are included in the Github repo. It is my hope this library will be
% useful in labs that already have a large investment in MATLAB, small
% applications where TensorFlow would be overkill, and/or pedagogical
% purposes. While ease of understanding is of course subjective, I think
% this library is easier to understand than some of the other ones out
% there. 
%
% This software is provided with ABSOLUTELY NO WARRANTIES.
classdef TinyBoltzmann    
    properties
        layers
        prevLayers
        weights
        alpha
        dalphadt
        trimThreshold
        minAlpha
        useSoftmaxTop
        verbose
        transferFunctions
    end
    
    methods
        function obj = TinyBoltzmann(dimVec, alpha, d_alpha_dt, trimThreshold)
            % By default, the RBM uses the logistic function for all layers
            % except the top; which uses softmax. If all you want to do is
            % set the top layer to logistic, simply set useSoftMaxTop to
            % false. If you want to do something else, use
            % setTransferFunctions() below.
            obj.layers = {};
            obj.weights = {};
            for idx = 1:numel(dimVec)
                if idx < numel(dimVec)
                    obj.weights{idx} = randn(dimVec(idx), dimVec(idx+1));
                end
                obj.layers{idx} = zeros(dimVec(idx), 1);
                obj.prevLayers{idx} = zeros(dimVec(idx), 1);
            end
            obj.alpha = alpha;
            obj.dalphadt = d_alpha_dt;
            obj.trimThreshold = trimThreshold;
            obj.minAlpha = 3.3E-4;
            obj.useSoftmaxTop = true;
            obj.verbose = false;
            obj.transferFunctions = {};
        end
        
        function obj = setTransferFunctions(obj, transferFunctionHandles)
            % By default, the RBM uses the logistic function for all layers
            % except the top; which uses softmax. If all you want to do is
            % set the top layer to logistic, simply set useSoftMaxTop to
            % false. If you want to do something else, use this function.
            %
            % transferFunctionHandles should be a cell array, containing
            % exactly 1 function handle for each layer in your RBM. 
            obj.transferFunctions = transferFunctionHandles;
        end
        
        function obj = projectLowerToHigher(obj, lowerNum, randSample)
            obj.layers{lowerNum+1} = transpose(obj.layers{lowerNum}' * obj.weights{lowerNum});
            if obj.verbose
                disp('upper layer shape:')
                disp(size(obj.layers{lowerNum+1}))
            end
            if numel(obj.transferFunctions) == numel(obj.layers)
                obj.layers{lowerNum+1} = arrayfun(obj.transferFunctions{lowerNum+1}, obj.layers{lowerNum+1});
            elseif lowerNum+1 == numel(obj.layers) && obj.useSoftmaxTop
                obj.layers{lowerNum+1} = arrayfun(@softmax, obj.layers{lowerNum+1});
            else
                obj.layers{lowerNum+1} = arrayfun(@sigmoid, obj.layers{lowerNum+1});
            end
            if nargin == 3 && randSample
                obj.layers{lowerNum+1} = arrayfun(@(x) x < rand(), obj.layers{lowerNum+1});
            end
        end
        
        
        function obj = projectHigherToLower(obj, lowerNum, randSample)
            obj.layers{lowerNum} = transpose(obj.layers{lowerNum+1}' * transpose(obj.weights{lowerNum}));
            if obj.verbose
                disp('lower layer shape:')
                disp(size(obj.layers{lowerNum}))
            end
            if numel(obj.transferFunctions) == numel(obj.layers)
                obj.layers{lowerNum} = arrayfun(obj.transferFunctions{lowerNum}, obj.layers{lowerNum});
            else
                obj.layers{lowerNum} = arrayfun(@sigmoid, obj.layers{lowerNum});
            end
            if nargin == 3 && randSample
                obj.layers{lowerNum} = arrayfun(@(x) rand() < x, obj.layers{lowerNum});
            end
        end

        
        function obj = trimWeights(obj, lowerNum)
            % Sometimes it is useful to trim weights that are very small,
            % such as 0.05. 
            obj.weights{lowerNum} = arrayfun(@(x) x * (abs(x) > obj.trimThreshold), obj.weights{lowerNum});
        end
        
        
        function y = learn(obj, lowerNum)
            % It should be okay to use this for all layers, with the
            % possible exception of guided training of the top layer. You can 
            % still use this for unguided training of the top layer,
            % however.
            %
            % If your transfer function is sufficiently exotic
            % (translation: weird), you may need to write your own learn
            % function. But you shouldn't need to do that if you're using
            % the logistic function, which is the default for all layers
            % except the top.
            % 
            % For guided training of the top layer, use learnTop() below. 
            errLower = obj.prevLayers{lowerNum} - obj.layers{lowerNum};
            errUpper = obj.prevLayers{lowerNum+1} - obj.layers{lowerNum+1};
            outerProduct = errLower * errUpper';
            if obj.verbose
                disp('Outer Product Size: ')
                disp(size(outerProduct));
            end
            y = outerProduct * obj.alpha;
        end
        
        function y = learnTop(obj, correctLabel)
            % Use this to train the top layer. This assumes you want to use
            % your RBM as a classifier. 
            %
            % When the top layer uses softmax, think of it like an LED that
            % lights up to show you which category the stimulus fits into. 
            %
            % correctLabel should be a column vector showing which
            % "bucket" the bottom layer best fits into.
            errLower = obj.prevLayers{end-1} - obj.layers{end-1};
            % TODO: Try the following line with and without normalizeMu0Sigma1():
            errUpper = correctLabel - obj.layers{end};
            outerProduct = errLower * errUpper';
            if obj.verbose
                disp('Top Layer Outer Product Size: ')
                disp(size(outerProduct));
            end
            y = outerProduct * obj.alpha;
            
        end
        
        
        function obj = pretrainLayer(obj, lowerNum, trainingSet, innerIterations, enableRandSamp)
            % Makes a single pass through your training data.
            % trainingSet must be a cell array.
            % innerIterations is the number of bounces you want for each
            % training specimen. 
            % lowerNum is the number of the lower of the 2 layers being
            % trained.
            % Note this function normalizes your data so that it has mu=0,
            % sigma=1.
            outerIterations = numel(trainingSet);
            for m = 1:outerIterations
                obj.layers{lowerNum} = normalizeMu0Sigma1(reshape(trainingSet{m}, size(obj.layers{lowerNum})));
                obj = obj.projectLowerToHigher(lowerNum, 0);
                obj.prevLayers{lowerNum} = obj.layers{lowerNum};
                obj.prevLayers{lowerNum+1} = obj.layers{lowerNum+1};
                for n = 1:innerIterations
                    obj = obj.projectHigherToLower(lowerNum, enableRandSamp);
                    obj = obj.projectLowerToHigher(lowerNum, enableRandSamp);
                end
                obj.weights{lowerNum} = obj.weights{lowerNum} + obj.learn(lowerNum);
                obj.alpha = obj.alpha * obj.dalphadt;
            end
        end
        
        
        function obj = pretrainNTimes(obj, lowerNum, trainingSet, innerIterations, outerIterations, enableRandSamp)
            for i = 1:outerIterations
                disp(sprintf('Starting pass %06d', i))
                obj = obj.pretrainLayer(lowerNum, trainingSet, innerIterations, enableRandSamp);
            end
        end
        
        function obj = pingPongTest(obj, lowerNum, oneImg, numBounce)
            % This function allows you to visualize the quality of a
            % reconstruction. After running this, run:
            %       >>  heatmap(reshape(uut.layers{1}, <img_height>, <img_width>))
            obj.layers{lowerNum} = normalizeMu0Sigma1(reshape(oneImg, size(obj.layers{lowerNum})));
            obj = obj.projectLowerToHigher(lowerNum, 0);
            for i = 1:numBounce
                obj = obj.projectHigherToLower(lowerNum, 0);
                obj = obj.projectLowerToHigher(lowerNum, 0);
%                if obj.alpha < obj.minAlpha
%                    disp('alpha is really tiny. Stopping training.')
%                    break;
%                end
            end
        end
    end
end


