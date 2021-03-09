import axios from 'axios';
import { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
import PriceCard from './PriceCard';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d[0])
    const labels = data.map(d => d[1]);
    console.log(labels)
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.tensor2d(inputData.map(d => d[0]), [inputData.length, 1])
    // const xs = tf.tens(0, 1, 100);
    const preds = model.predict(xs.reshape([xs.size, 1]));
    // const labelMax = preds.max();
    // const labelMin = preds.min();

    const inputMax = xs.max();
    const inputMin = xs.min();

    const unNormXs = xs.sub(inputMin).div(inputMax.sub(inputMin));
    const unNormPreds = preds.sub(labelMin).div(labelMax.sub(labelMin));

    // const unNormXs = xs
    //   .mul(inputMax.sub(inputMin))
    //   .add(inputMin);

    // const unNormPreds = preds
    //   .mul(labelMax.sub(labelMin))
    //   .add(labelMin);
    console.log(unNormXs)
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  });

  const originalPoints = inputData.map(d => ({
    x: d[0], y: d[1],
  }));

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Réanimation',
      yLabel: 'Hospitalisation',
      height: 300
    }
  );
}

function App() {
  async function run(data) {
    const train = data.slice(0, 300);
    const test = data.slice(301, 352);

    const values = data.map(d => ({
      x: d[0],
      y: d[1],
    }));

    tfvis.render.scatterplot(
      { name: 'Réanimation' },
      { values },
      {
        xLabel: 'Réanimation',
        yLabel: 'Hospitalisation',
        height: 300
      }
    );

    // We want to predict the column "medv", which represents a median value of
    // a home (in $1000s), so we mark it as a label.
    const model = createModel();
    const tensorData = convertToTensor(train);

    const { inputs, labels } = tensorData;
    // console.log(data)

    // Train the model
    await trainModel(model, inputs, labels);
    testModel(model, test, tensorData);


    // Fit the model using the prepared Dataset

    // await model.fit(input, label, {
    //   epochs: 250, callbacks: {
    //     onEpochEnd: async (epoch, logs) => {
    //       console.log(logs);
    //     }
    //   }
    // });
    // console.log('finish')
    // const pre = model.predict(tf.tensor2d([20], [1, 1])).dataSync()
    // console.log(pre)
  }
  const [ticker, setTicker] = useState({
    low: 0,
    high: 0,
    last: 0,
  });



  useEffect(() => {
    async function getDogecoinPrice() {
      const result = await axios.get(
        'https://coronavirusapi-france.now.sh/AllDataByDepartement?Departement=Bas-Rhin'
      );
      const hospitalises = result.data.allDataByDepartement.map((res) => [
        // new Date(res.date).getTime(),
        res.reanimation,
        res.hospitalises,
      ]).filter((t) => t[0] !== undefined && t[1] !== undefined)

      run(hospitalises)


    }
    getDogecoinPrice();
  }, []);

  return (
    <div className="App">
      <img src={logo} width={150} height={150} alt="Dogecoin Logo" />
      <h1 className="title">Live Dogecoin Price</h1>
      <h5 className="subtitle">Dogecoin To The Moon 🚀🌕</h5>
      <div className="prices-container">
        {/* <PriceCard type="low" price={ticker.low} />
        <PriceCard type="high" price={ticker.high} />
        <PriceCard type="current" price={ticker.last} /> */}
      </div>
      <p>
        Dogecoin price updated every 10 seconds seconds from{' '}
        <a href="https://wazirx.com/">WazirX API</a>
      </p>
    </div>
  );
}

export default App;
