classdef TinyBoltzmann
    
    properties
        layers
        prevLayers
        weights
        alpha
        dalphadt
        trimThreshold
        minAlpha
    end
    
    methods
        function obj = TinyBoltzmann(dimVec, alpha, d_alpha_dt, trimThreshold)
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
        end
        
        
        function obj = projectLowerToHigher(obj, lowerNum, randSample)
            obj.layers{lowerNum+1} = transpose(obj.layers{lowerNum}' * obj.weights{lowerNum});
            %disp('upper layer shape:')
            %disp(size(obj.layers{lowerNum+1}))
            obj.layers{lowerNum+1} = arrayfun(@sigmoid, obj.layers{lowerNum+1});
            if nargin == 3 && randSample
                obj.layers{lowerNum+1} = arrayfun(@(x) x < rand(), obj.layers{lowerNum+1});
            end
        end
        
        
        function obj = projectHigherToLower(obj, lowerNum, randSample)
            obj.layers{lowerNum} = transpose(obj.layers{lowerNum+1}' * transpose(obj.weights{lowerNum}));
            %disp('lower layer shape:')
            %disp(size(obj.layers{lowerNum}))
            obj.layers{lowerNum} = arrayfun(@sigmoid, obj.layers{lowerNum});
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
            errLower = obj.prevLayers{lowerNum} - obj.layers{lowerNum};
            errUpper = obj.prevLayers{lowerNum+1} - obj.layers{lowerNum+1};
            outerProduct = errLower * errUpper';
            %disp('Outer Product Size: ')
            %disp(size(outerProduct));
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
            obj.layers{lowerNum} = normalizeMu0Sigma1(reshape(oneImg, size(obj.layers{lowerNum})));
            obj = obj.projectLowerToHigher(lowerNum, 0);
            for i = 1:numBounce
                obj = obj.projectHigherToLower(lowerNum, 0);
                obj = obj.projectLowerToHigher(lowerNum, 0);
                if obj.alpha < obj.minAlpha
                    disp('alpha is really tiny. Stopping training.')
                    break;
                end
            end
        end
    end
end


