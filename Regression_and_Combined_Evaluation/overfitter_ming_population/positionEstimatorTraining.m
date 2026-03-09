function modelParameters = positionEstimatorTraining(trainingData)
% Direction from POPULATION VECTOR (tuning); then per-direction velocity regression.
% Uses smoothed firing rate; no LDA — direction = angle of sum(rate_n * prefDir_n).

step = 20;
bufferBinsLong  = 20;
bufferBinsShort = 5;
initTime = 300;
dirWindowEnd = 320;
smoothWinLen = 5;
lambda = 15;
avgTrajWeight = 0.4;

numTrials  = size(trainingData,1);
numNeurons = size(trainingData(1,1).spikes,1);

smoothKernel = ones(1, smoothWinLen) / smoothWinLen;

% ---------- 1) Per-neuron mean rate per direction (early window) ----------
dirMeanRates = zeros(numNeurons, 8);   % neurons x directions
for d = 1:8
    F = [];
    for i = 1:numTrials
        raw = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        Tend = min(size(cleaned,2), dirWindowEnd);
        feat = sum(cleaned(:,1:Tend), 2)';   % 1 x 98
        F = [F; feat];
    end
    dirMeanRates(:, d) = mean(F, 1)';   % 98 x 1
end

% ---------- 2) Preferred direction per neuron (population vector) ----------
% Direction d has angle (d-1)*pi/4; unit vector = [cos; sin].
angles = (0:7)' * (pi/4);
unitVecs = [cos(angles), sin(angles)];   % 8 x 2
% Preferred = weighted sum of directions by mean rate, then normalize.
prefVec = zeros(numNeurons, 2);
for n = 1:numNeurons
    w = dirMeanRates(n, :);   % 1 x 8
    p = w * unitVecs;         % 1 x 2
    nrm = norm(p);
    if nrm > 1e-8
        prefVec(n, :) = p / nrm;
    else
        prefVec(n, :) = [1 0];   % fallback
    end
end

% ---------- 3) Per-direction VELOCITY regression (multi-scale buffer) ----------
linearModelVx = cell(1,8);
linearModelVy = cell(1,8);
featMean = cell(1,8);
featStd  = cell(1,8);

for d = 1:8
    X = [];
    Yvel = [];
    for i = 1:numTrials
        raw  = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        pos  = trainingData(i,d).handPos(1:2,:);
        T    = size(cleaned,2);

        for t = (initTime+step):step:T
            startLong  = t - bufferBinsLong  + 1;
            startShort = t - bufferBinsShort + 1;
            if startLong  < 1, startLong  = 1; end
            if startShort < 1, startShort = 1; end
            featLong  = sum(cleaned(:, startLong:t), 2)';
            featShort = sum(cleaned(:, startShort:t), 2)';
            feat = [featShort, featLong];
            vel = (pos(:,t) - pos(:,t-step))';
            X = [X; feat];
            Yvel = [Yvel; vel];
        end
    end

    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig < 1e-6) = 1;
    Xn = (X - mu) ./ sig;
    Xb = [Xn ones(size(Xn,1),1)];
    featMean{d} = mu;
    featStd{d}  = sig;

    I = eye(size(Xb,2)); I(end,end) = 0;
    wVx = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,1));
    wVy = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,2));
    linearModelVx{d} = wVx;
    linearModelVy{d} = wVy;
end

% ---------- 4) Direction-specific average trajectory ----------
avgTraj = cell(1,8);
for d = 1:8
    Tmax = 0;
    for i = 1:numTrials
        Tmax = max(Tmax, size(trainingData(i,d).handPos, 2));
    end
    sumPos = zeros(2, Tmax);
    count  = zeros(1, Tmax);
    for i = 1:numTrials
        pos = trainingData(i,d).handPos(1:2,:);
        T_i = size(pos, 2);
        sumPos(:, 1:T_i) = sumPos(:, 1:T_i) + pos;
        count(1:T_i) = count(1:T_i) + 1;
    end
    count(count == 0) = 1;
    avgTraj{d} = sumPos ./ repmat(count, 2, 1);
end

% Pack
modelParameters.step = step;
modelParameters.bufferSizeLong  = bufferBinsLong;
modelParameters.bufferSizeShort = bufferBinsShort;
modelParameters.dirWindowEnd = dirWindowEnd;
modelParameters.smoothWinLen = smoothWinLen;
modelParameters.smoothKernel = smoothKernel;
modelParameters.prefVec = prefVec;   % 98 x 2 (population vector weights)
modelParameters.linearModelVx = linearModelVx;
modelParameters.linearModelVy = linearModelVy;
modelParameters.featMean = featMean;
modelParameters.featStd  = featStd;
modelParameters.avgTraj = avgTraj;
modelParameters.avgTrajWeight = avgTrajWeight;
modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
end
