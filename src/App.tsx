import {
  Anchor,
  Box,
  Button,
  Container,
  Divider,
  Grid,
  Group,
  Text,
  ThemeIcon,
  useMantineTheme,
} from '@mantine/core';
import { IconBrandGithub } from '@tabler/icons';
import * as tf from '@tensorflow/tfjs';
import { useRef, useState } from 'react';
import { MnistData } from './classes/mnist-data';

function App() {
  const theme = useMantineTheme();
  const [logs, setLogs] = useState<string | null>('training required!');
  const [trainedData, setTrainedData] = useState<any>();
  const [trainedModel, setTrainedModel] = useState<any>();
  const [prediction, setPrediction] = useState<number | null>(null);
  const canvasRef = useRef();

  let model: any;
  let data: any;
  function createModel() {
    setLogs('creating model...');
    model = tf.sequential();
    setLogs('model created.');

    setLogs('creating layers..');
    model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling',
      })
    );

    model.add(
      tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2],
      })
    );

    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax',
      })
    );

    setLogs('layers created.');

    setLogs('compiling...');
    model.compile({
      optimizer: tf.train.sgd(0.15),
      loss: 'categoricalCrossentropy',
    });

    setLogs('finished compiling.');

    setTrainedModel(model);
  }

  async function load() {
    setLogs('loading data...');
    data = new MnistData();
    // setData(new MnistData());
    await data.load();
    setLogs('data loaded succesfully');
  }

  const BATCH_SIZE: number = 64;
  const TRAIN_BATCHES: number = 150;

  async function train() {
    setLogs('training started...');
    for (let i = 0; i < TRAIN_BATCHES; i++) {
      // eslint-disable-next-line no-loop-func
      const batch = tf.tidy(() => {
        const batch = data.nextTrainBatch(BATCH_SIZE);
        batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
        return batch;
      });

      await model.fit(batch.xs, batch.labels, {
        batchSize: BATCH_SIZE,
        epochs: 1,
      });

      tf.dispose(batch);

      await tf.nextFrame();
    }

    setTrainedData(data);
    setTrainedModel(model);
    setLogs('training complete');
  }

  async function predict(batch: any) {
    tf.tidy(() => {
      const input_value = Array.from(batch.labels.argMax(1).dataSync()) as any;
      const output = trainedModel.predict(batch.xs.reshape([-1, 28, 28, 1]));
      const prediction_value = Array.from(output.argMax(1).dataSync()) as any;
      const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);
      const canvas = document.getElementById('prediction-canvas')!;
      draw(image.flatten(), canvas);

      if (prediction_value - input_value === 0) {
        setPrediction(prediction_value[0]);
        setPrediction(prediction_value[0]);
        setLogs('recognition success');
      } else {
        setPrediction(prediction_value[0] || null);
        setLogs('recognition failed');
      }
    });
  }

  function draw(image: any, canvas: any) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  async function handleTest() {
    createModel();
    await load();
    await train();
  }

  async function handlePredict() {
    const batch = trainedData.nextTestBatch(1);
    await predict(batch);
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100%',
        position: 'relative',
        backgroundColor: theme.colors.dark[9],
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Box sx={{ width: '100%' }}>
        <Container
          size='xs'
          sx={{
            borderRadius: theme.radius.md,
            backgroundColor: theme.colors.dark[7],
          }}
          p={'lg'}
        >
          <Box py={4}>
            <Text weight='bold' mb={3}>
              Welcome to Digits Recognizer V1.0
            </Text>
            <Text size='sm' color='dimmed'>
              To use this tool, you first have start a training process before
              trying to predict a digit, press the "Start Training" button to
              start.
            </Text>
            <Box
              style={{
                marginTop: 8,
                borderRadius: theme.radius.md,
                width: '100%',
                padding: 12,
                backgroundColor: theme.colors.dark[9],
              }}
            >
              <Text
                size='sm'
                weight='500'
                color={
                  logs === 'training complete' || logs === 'recognition success'
                    ? 'green'
                    : logs === 'training required!' ||
                      logs === 'recognition failed'
                    ? 'red'
                    : 'orange'
                }
              >
                <Text
                  component='span'
                  color='dimmed'
                  style={{ marginInlineEnd: 4 }}
                >
                  Logs:
                </Text>{' '}
                {logs}
              </Text>
            </Box>
          </Box>

          <>
            {(prediction === 0 || prediction) && (
              <Divider my={12} variant='dashed' />
            )}
            <Box className='prediction-div'>
              <Grid>
                <Grid.Col span={4}>
                  {(prediction === 0 || prediction) && (
                    <Box
                      style={{
                        borderRadius: theme.radius.md,
                        width: '100%',
                        height: '100%',
                        padding: 12,
                        backgroundColor: theme.colors.dark[9],
                      }}
                    >
                      <Box
                        style={{
                          width: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          paddingInlineEnd: 8,
                        }}
                      >
                        <Text
                          component='span'
                          color='dimmed'
                          style={{
                            marginInlineEnd: 4,
                          }}
                          size='xs'
                        >
                          Prediction Value
                        </Text>
                        <Text size='md' weight='500'>
                          {prediction}
                        </Text>
                      </Box>
                    </Box>
                  )}
                </Grid.Col>
                <Grid.Col span={8}>
                  <canvas
                    ref={canvasRef.current}
                    id='prediction-canvas'
                    style={{
                      width: '100%',
                      borderRadius: 16,
                    }}
                  ></canvas>
                </Grid.Col>
              </Grid>
            </Box>
          </>

          <Divider my={16} variant='dashed' />
          <Box>
            <Text size='sm' weight='500' mb={-4}>
              Crafted by Abderraouf Zine
            </Text>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Text size='xs' color='dimmed'>
                MIT Licensed (Open Source)
              </Text>
              <Group spacing={4}>
                <Text size='xs' color='dimmed'>
                  See full code on
                </Text>
                <Anchor
                  href='https://github.com/rofazayn/handwriting-recognition'
                  target='_blank'
                  sx={{ color: theme.colors.gray[4] }}
                >
                  <Group spacing={3}>
                    <Text size='xs' underline>
                      GitHub
                    </Text>
                    <IconBrandGithub width='16' />
                  </Group>
                </Anchor>
              </Group>
            </Box>
          </Box>

          <Divider my={16} variant='dashed' />
          <Group>
            <Button
              onClick={async () => await handleTest()}
              color='indigo'
              radius='md'
              disabled={logs !== 'training required!'}
            >
              Start Training the Model
            </Button>

            <Button
              onClick={async () => await handlePredict()}
              color='indigo'
              radius='md'
            >
              Predict
            </Button>
          </Group>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
