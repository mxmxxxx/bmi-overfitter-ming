function modelParameters = positionEstimatorTraining(trainingData)
% Self-contained: Manual LDA direction classification + per-direction ridge regression
% No extra .m dependencies, compatible with testFunction_for_students_MTb

% -------- settings --------
step = 20;              % must match test script
bufferBins = 10;        % regression feature window length in bins (~200ms)
initTime = 300;         % start regression after 300ms
dirWindowEnd = 320;     % use early window [1:320] for direction classification

lambda = 100;           % ridge strength (tune: 1,10,100,1000...)
ldaReg = 1e-2;          % LDA covariance regularization (tune: 1e-4..1e-1)

[numTrials, numDirs] = size(trainingData);
numNeurons = size(trainingData(1,1).spikes, 1);

% =========================================================
% 1) Build early-window features for LDA classification
% =========================================================
N = numTrials * numDirs;
Xcls = zeros(N, numNeurons);
ycls = zeros(N, 1);

idx = 1;
for i = 1:numTrials
    for d = 1:numDirs
        spk = trainingData(i,d).spikes;                 % 98 x T
        Tend = min(size(spk,2), dirWindowEnd);
        Xcls(idx,:) = sum(spk(:,1:Tend), 2)';           % 1 x 98 (counts)
        ycls(idx) = d;
        idx = idx + 1;
    end
end

% z-score features (important for LDA stability)
mu = mean(Xcls, 1);
sigma = std(Xcls, 0, 1) + eps;
Xn = (Xcls - mu) ./ sigma;

% compute class means in normalized space
muClass = zeros(numDirs, numNeurons);
for d = 1:numDirs
    muClass(d,:) = mean(Xn(ycls==d, :), 1);
end

% pooled covariance (shared across classes)
Sigma = cov(Xn);                              % 98x98
Sigma = Sigma + ldaReg * eye(numNeurons);     % regularize
invSigma = inv(Sigma);

% LDA scoring: score_d(x) = x*invSigma*mu_d' - 0.5*mu_d*invSigma*mu_d'
Wlda = invSigma * muClass';                   % 98 x 8
clda = -0.5 * sum(muClass .* (muClass*invSigma), 2);  % 8 x 1

% =========================================================
% 2) Train per-direction ridge regression models (buffered)
% =========================================================
wX = cell(1, numDirs);
wY = cell(1, numDirs);

for d = 1:numDirs
    X = [];
    Y = [];

    for i = 1:numTrials
        spikes = trainingData(i,d).spikes;            % 98 x T
        pos    = trainingData(i,d).handPos(1:2,:);    % 2 x T
        T      = size(spikes,2);

        for t = (initTime+step):step:T
            startBin = t - bufferBins + 1;
            if startBin < 1
                startBin = 1;
            end

            feat = sum(spikes(:, startBin:t), 2)';    % 1 x 98
            X = [X; feat];
            Y = [Y; pos(:,t)'];
        end
    end

    Xb = [X ones(size(X,1),1)];   % bias column

    % ridge (do not regularize bias term)
    I = eye(size(Xb,2));
    I(end,end) = 0;

    wX{d} = (Xb'*Xb + lambda*I) \ (Xb'*Y(:,1));
    wY{d} = (Xb'*Xb + lambda*I) \ (Xb'*Y(:,2));
end

% =========================================================
% pack everything
% =========================================================
modelParameters.step = step;
modelParameters.bufferBins = bufferBins;
modelParameters.initTime = initTime;

% LDA params
modelParameters.dirWindowEnd = dirWindowEnd;
modelParameters.mu = mu;
modelParameters.sigma = sigma;
modelParameters.Wlda = Wlda;
modelParameters.clda = clda;

% regression params
modelParameters.wX = wX;
modelParameters.wY = wY;

% cache predicted direction per trialId during decoding
modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
end