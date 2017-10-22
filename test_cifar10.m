%% https://au.mathworks.com/help/vision/examples/object-detection-using-deep-learning.html

% Download CIFAR-10 data to a temporary directory
% cifar10Data = tempdir;
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
% 
% helperCIFAR10Data.download(url, cifar10Data);

%% Load the CIFAR-10 training and test data. Note that it automatically
% appends the name of the folder... hence blank input arg. comment out
% after loading to save time
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load('');

%% Display a few of the training images, resizing them for display.
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

%% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize);

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

%% Create middle layers
middleLayers = [

% The first convolutional layer has a bank of 32 5x5x3 filters. A
% symmetric padding of 2 pixels is added to ensure that image borders
% are included in the processing. This is important to avoid
% information at the borders being washed away too early in the
% network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

% Note that the third dimension of the filter can be omitted because it
% is automatically deduced based on the connectivity of the network. In
% this case because this layer follows the image layer, the third
% dimension must be 3 to match the number of channels in the input
% image.

% Next add the ReLU layer:
reluLayer()

% Follow it with a max pooling layer that has a 3x3 spatial pooling area
% and a stride of 2 pixels. This down-samples the data dimensions from
% 32x32 to 15x15.
maxPooling2dLayer(3, 'Stride', 2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

]

%% Create final layers
numImageCategories = 10;
finalLayers = [

% Add a fully connected layer with 64 output neurons. The output size of
% this layer will be an array with a length of 64.
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce 10 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.

fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
]

%% Make all the layers

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%Initialize the first convolutional layer weights using normally distributed random numbers with standard deviation of 0.0001. This helps improve the convergence of training.
layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%% Train CNN Using CIFAR-10 Data
%Now that the network architecture is defined, it can be trained using the CIFAR-10 training data. First, set up the network training algorithm using the trainingOptions function. The network training algorithm uses Stochastic Gradient Descent with Momentum (SGDM) with an initial learning rate of 0.001. During training, the initial learning rate is reduced every 8 epochs (1 epoch is defined as one complete pass through the entire training data set). The training algorithm is run for 40 epochs.
%Note that the training algorithm uses a mini-batch size of 128 images. If using a GPU for training, this size may need to be lowered due to memory constraints on the GPU.

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

%% Select network training
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining = false;

if doTraining
    % Train a network.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
    % Load pre-trained detector for the example.
    load('rcnnStopSigns.mat','cifar10Net')
end

%%  After the network is trained, it should be validated to ensure that training was successful. First, a quick visualization of the first convolutional layer's filter weights can help identify any immediate issues with training.

% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale and resize the weights for better visualization
w = mat2gray(w);
w = imresize(w, [100 100]);

figure
montage(w)

% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)

%% TESTING FOR R-CNN

%Now that the network is working well for the CIFAR-10 classification task, the transfer learning approach can be used to fine-tune the network for stop sign detection.
%Start by loading the ground truth data for stop signs.

% Load the ground truth data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

% Update the path to the image files to match the local file system
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

% Display a summary of the ground truth data
summary(stopSignsAndCars)

% Only keep the image file names and the stop sign ROI labels
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Display one training image and the ground truth bounding boxes
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I, 'Rectangle', stopSigns.stopSign{1}, 'stop sign', 'LineWidth', 8);

figure
imshow(I)

%% Select network training

% A trained detector is loaded from disk to save time when running the
% example. Set this flag to true to train the detector.
doTraining = true;

if doTraining

    % Set training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);

    % Train an R-CNN object detector. This will take several minutes.
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
else
    % Load pre-trained network for the example.
    load('rcnnStopSigns.mat','rcnn')
end

%% Test R-CNN

% Read test image
testImage = imread('stopSignTest.jpg');

% Detect stop signs
[bboxes, score, label] = detect(rcnn, testImage, 'MiniBatchSize', 128)

% The R-CNN object detect method returns the object bounding boxes, a detection score, and a class label for each detection. The labels are useful when detecting multiple objects, e.g. stop, yield, or speed limit signs. The scores, which range between 0 and 1, indicate the confidence in the detection and can be used to ignore low scoring detections.
% Display the detection results
[score, idx] = max(score);

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage)

%% https://au.mathworks.com/help/vision/ref/trainrcnnobjectdetector.html

%% https://au.mathworks.com/help/nnet/ug/layers-of-a-convolutional-neural-network.html
%% https://au.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html
%% https://au.mathworks.com/help/nnet/examples/train-a-convolutional-neural-network-for-regression.html
