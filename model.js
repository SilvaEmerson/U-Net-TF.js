const tf = require("@tensorflow/tfjs-node");

const IMAGE_INPUT = [512, 512];

const buildInput = (x_dim, y_dim, channels = 1) =>
  tf.input({ shape: [x_dim, y_dim, channels] });

const genConv2D = (filters, kernelDim = 3) =>
  tf.layers.conv2d({
    filters: filters,
    kernelSize: [kernelDim, kernelDim],
    activation: "relu",
    padding: "same",
    kernelInitializer: "heNormal"
  });

const geMaxPool2D = poolDim =>
  tf.layers.maxPooling2d({
    poolSize: [poolDim, poolDim]
  });

const genUp2D = (kernelDim = 2) =>
  tf.layers.upSampling2d({
    size: [kernelDim, kernelDim],
    dataFormat: "channelsLast"
  });

//First part (down climb)
const input = buildInput(...IMAGE_INPUT);
const conv1 = genConv2D(64).apply(input);
const conv2 = genConv2D(64).apply(conv1);
const pool1 = geMaxPool2D(2).apply(conv2);
const conv3 = genConv2D(128).apply(pool1);
const conv4 = genConv2D(128).apply(conv3);
const pool2 = geMaxPool2D(2).apply(conv4);
const conv5 = genConv2D(256).apply(pool2);
const conv6 = genConv2D(256).apply(conv5);
const pool3 = geMaxPool2D(2).apply(conv6);
const conv7 = genConv2D(512).apply(pool3);
const conv8 = genConv2D(512).apply(conv7);
const pool4 = geMaxPool2D(2).apply(conv8);
const conv9 = genConv2D(1024).apply(pool4);
const conv10 = genConv2D(1024).apply(conv9);
const up1 = genUp2D().apply(conv10);
const merge1 = tf.layers.concatenate({ axis: 3 }).apply([up1, conv8]);

//Second part (up climb)
const conv11 = genConv2D(512).apply(merge1);
const conv12 = genConv2D(512).apply(conv11);
const up2 = genUp2D().apply(conv12);
const merge2 = tf.layers.concatenate({ axis: 3 }).apply([up2, conv6]);

const conv13 = genConv2D(256).apply(merge2);
const conv14 = genConv2D(256).apply(conv13);
const up3 = genUp2D().apply(conv14);
const merge3 = tf.layers.concatenate({ axis: 3 }).apply([up3, conv4]);

const conv15 = genConv2D(128).apply(merge3);
const conv16 = genConv2D(128).apply(conv15);
const up4 = genUp2D().apply(conv16);
const merge4 = tf.layers.concatenate({ axis: 3 }).apply([up4, conv2]);

const conv17 = genConv2D(64).apply(merge4);
const conv18 = genConv2D(64).apply(conv17);
const conv19 = tf.layers
  .conv2d({
    kernelSize: [1, 1],
    activation: "sigmoid",
    filters: 1,
    kernelInitializer: "heNormal",
    padding: "same"
  })
  .apply(conv18);

const model = tf.model({ inputs: input, outputs: conv19 });

// Sigmoid cross-entropy is also known as binary cross-entropy
model.compile({
  loss: tf.losses.sigmoidCrossEntropy,
  optimizer: tf.train.adam(1e-4),
  metrics: tf.metrics.binaryCrossentropy
});

model.summary();
