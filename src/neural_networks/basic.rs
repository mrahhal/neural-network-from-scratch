use std::{
    cell::{Cell, RefCell},
    f64::EPSILON,
};

use nalgebra::{Dynamic, MatrixSlice, U1};

use crate::utils::{rnd_normal, softmax, Matrix, Vector};

/// A basic deep neural network.
///
/// I: The number of neurons in the input layer.
/// O: The number of neurons in the output layer.
pub struct Network<const I: usize, const O: usize> {
    learning_rate: f64,

    /// All layers after the input layer (i.e hidden layers + output layer).
    layers: Vec<Layer>,
}

fn create_layer(
    prev_neurons_count: usize,
    neurons_count: usize,
    activator_creator: impl Fn() -> Box<dyn Activator>,
) -> Layer {
    let mut neurons = Vec::with_capacity(neurons_count);
    for _ in 0..neurons_count {
        let neuron = Neuron::new(
            generate_random_neuron_params(prev_neurons_count),
            activator_creator(),
        );

        neurons.push(neuron);
    }

    Layer::new(neurons)
}

impl<const I: usize, const O: usize> Network<I, O> {
    pub fn new() -> Self {
        let hidden_layers = [200, 80];
        let hidden_layers_count = hidden_layers.len();

        // + 1 output layer
        let mut layers = Vec::with_capacity(hidden_layers_count + 1);
        let mut prev_neurons_count = I;
        for i in 0..hidden_layers_count {
            let neurons_count = hidden_layers[i];
            layers.push(create_layer(prev_neurons_count, neurons_count, || {
                Box::new(ReLUActivator)
            }));
            prev_neurons_count = neurons_count;
        }

        layers.push(create_layer(prev_neurons_count, O, || {
            Box::new(NoopActivator)
        }));

        Self {
            learning_rate: 0.01,
            layers,
        }
    }

    pub fn learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Run through the network and produce an output.
    pub fn run(&self, input: &[f64]) -> Vec<f64> {
        if I != input.len() {
            panic!("Mismatch of inputs length.")
        }

        let mut prev_activs = Vector::from_vec(input.to_owned());
        for layer in &self.layers {
            prev_activs = layer.forward(prev_activs);
        }

        let output_activs_non_normalized = prev_activs;
        let softmaxed = softmax(&output_activs_non_normalized);

        softmaxed.data.into()
    }

    pub fn train(&mut self, samples: &Vec<(Vec<f64>, Vec<f64>)>) {
        let mut total_gradients: Option<SinglePassGradients> = None;
        let samples_count = samples.len() as f64;

        for (input, expected) in samples {
            let gradients = self.backward(input, expected);
            // TODO: refactor
            if total_gradients.is_none() {
                total_gradients.replace(gradients);
            } else {
                let total_gradients = total_gradients.as_mut().unwrap();
                for i in 0..gradients.weights.len() {
                    let w = &gradients.weights[i];
                    let w_t = &mut total_gradients.weights[i];

                    // TODO: optimize
                    for (rowi, row) in w.row_iter().enumerate() {
                        for (coli, v) in row.iter().enumerate() {
                            w_t[(rowi, coli)] += v;
                        }
                    }
                    // w.add_to(w_t, w_t);

                    let b = &gradients.biases[i];
                    let b_t = &mut total_gradients.biases[i];
                    for (i, v) in b.iter().enumerate() {
                        b_t[i] += v;
                    }
                    // b.add_to(b_t, b_t);
                }
            }
        }

        // Produce gradient mean.

        let mut total_gradients = total_gradients.unwrap();
        for w in &mut total_gradients.weights {
            for v in w.iter_mut() {
                // combining mean and step computation
                *v /= samples_count;
                *v *= -self.learning_rate;
            }
        }
        for b in &mut total_gradients.biases {
            for v in b.iter_mut() {
                // combining mean and step computation
                *v /= samples_count;
                *v *= -self.learning_rate;
            }
        }

        // Perform a gradient descent step.
        let descent = total_gradients;

        for (i, layer) in &mut self.layers.iter_mut().enumerate() {
            let layer_descent_w = &descent.weights[i];
            let layer_descent_b = &descent.biases[i];

            for (ni, n) in &mut layer.neurons.iter_mut().enumerate() {
                let n_descent_w = layer_descent_w.column(ni);
                n.params.weights += n_descent_w;
                n.params.bias += layer_descent_b[ni];
            }
        }
    }

    fn backward(&mut self, input: &[f64], expected: &[f64]) -> SinglePassGradients {
        // output is softmaxed
        let output = self.run(&input);

        // Don't need it because it's implicit in d_l_z below.
        // let loss = CCELoss::sample_loss(&expected, &output);

        // Calculate deriv of CCE + Softmax and produce a Matrix to start feeding the first layer.
        let mut d_l_z = Matrix::zeros(output.len(), 1);
        for i in 0..output.len() {
            d_l_z[i] = output[i] - expected[i];
        }

        // Start applying those to layers in back order.
        let mut prev_values = d_l_z;
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            prev_values = layer.backward(&prev_values);
        }

        // Collect gradients.
        let mut g_weights = vec![];
        let mut g_biases = vec![];

        for layer in &self.layers {
            let prev_layer_neuron_count = layer.neurons[0].params.weights.len();
            let curr_layer_neuron_count = layer.neurons.len();

            let mut g_layer_weights =
                Matrix::zeros(prev_layer_neuron_count, curr_layer_neuron_count);
            let mut g_layer_biases = Vec::with_capacity(curr_layer_neuron_count);

            for (i, n) in layer.neurons.iter().enumerate() {
                g_layer_weights.set_column(i, &n.gradient.as_ref().unwrap().weights);
                g_layer_biases.push(n.gradient.as_ref().unwrap().bias);
            }

            g_weights.push(g_layer_weights);
            g_biases.push(Vector::from_vec(g_layer_biases));
        }

        SinglePassGradients {
            weights: g_weights,
            biases: g_biases,
        }
    }

    #[allow(dead_code)]
    pub fn print_weights(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.params.weights);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_biases(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.params.bias);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_gradients(&self) {
        for layer in &self.layers {
            for n in &layer.neurons {
                println!("{}", n.gradient.as_ref().unwrap().weights);
                println!("{}", n.gradient.as_ref().unwrap().bias);
            }
        }
    }
}

struct SinglePassGradients {
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

/// Represents a layer in a network.
struct Layer {
    /// The neurons forming the layer.
    neurons: Vec<Neuron>,

    // /// The activator used in the layer.
    // activator: Box<dyn Activator>,
    /// The current run's activations of the preceding layer.
    /// Used when backpropagating.
    prev_activs: RefCell<Option<Vector>>,
}

impl Layer {
    fn new(neurons: Vec<Neuron>) -> Self {
        Layer {
            neurons,
            prev_activs: Default::default(),
        }
    }

    fn forward(&self, prev_activs: Vector) -> Vector {
        self.prev_activs.replace(Some(prev_activs));
        let prev_activs = self.prev_activs.borrow();
        let prev_activs = prev_activs.as_ref().unwrap();

        let mut activs = Vec::with_capacity(self.neurons.len());
        for n in &self.neurons {
            let activ = n.forward(prev_activs);
            activs.push(activ);
        }
        Vector::from_vec(activs)
    }

    // next_values includes a dot for W if this is a hidden layer.
    fn backward(&mut self, next_values: &Matrix) -> Matrix {
        let prev_activs = self.prev_activs.borrow();
        let prev_activs = prev_activs.as_ref().unwrap();

        let prev_layer_neuron_count = self.neurons[0].params.weights.len();
        let curr_layer_neuron_count = self.neurons.len();

        // Values for a single neuron will be organized in a column.
        let mut values_matrix = Matrix::zeros(prev_layer_neuron_count, curr_layer_neuron_count);
        for i in 0..self.neurons.len() {
            let neuron = &mut self.neurons[i];
            let next_values_for_neuron = next_values.row(i);
            let values = neuron.backward(prev_activs, next_values_for_neuron);
            values_matrix.set_column(i, &values);
        }

        values_matrix
    }
}

/// Represents a neuron in a layer.
struct Neuron {
    params: NeuronParams,
    /// Current activation of the neuron.
    z: Cell<f64>,
    activ: Cell<f64>,
    gradient: Option<NeuronGradient>,

    activator: Box<dyn Activator>,
}

impl Neuron {
    fn new(params: NeuronParams, activator: Box<dyn Activator>) -> Self {
        Self {
            params,
            z: Default::default(),
            activ: Default::default(),
            gradient: None,
            activator,
        }
    }

    /// Does a forward step.
    ///
    /// z = Σw.a + b
    fn forward(&self, prev_activs: &Vector) -> f64 {
        debug_assert!(self.params.weights.len() == prev_activs.len());

        let z = self.params.weights.dot(prev_activs) + self.params.bias;
        self.z.set(z);

        let activ = self.activator.activate(z);
        self.activ.set(activ);

        activ
    }

    /// Does a backward step, returning a vector of `g_z.W`.
    fn backward<'a, 'b>(
        &'a mut self,
        prev_activs: &Vector,
        next_values: MatrixSlice<'b, f64, U1, Dynamic, U1, Dynamic>,
    ) -> Vector {
        let z = self.z.get();

        // da/dz
        let d_a_z = self.activator.derivative(z);

        // dL/dz
        // next_values are `g_z.W` from the next layer.
        let g_z = d_a_z * next_values.sum();

        // dL/dw
        let g_w = prev_activs * g_z;

        // dL/db
        let g_b = g_z;

        self.gradient.replace(NeuronGradient {
            weights: g_w,
            bias: g_b,
        });

        g_z * &self.params.weights
    }
}

struct NeuronParams {
    weights: Vector,
    bias: f64,
}

struct NeuronGradient {
    weights: Vector,
    bias: f64,
}

trait Activator {
    fn activate(&self, z: f64) -> f64;

    fn derivative(&self, z: f64) -> f64;
}

/// Rectified linear unit ([ReLU]) activation.
///
/// [ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
struct ReLUActivator;

impl Activator for ReLUActivator {
    /// a(z) =
    ///     0 if z <= 0
    ///     z if z > 0
    fn activate(&self, z: f64) -> f64 {
        z.max(0.0)
    }

    /// a' =
    ///     0 if z <= 0
    ///     1 if z > 0
    fn derivative(&self, z: f64) -> f64 {
        if z > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// [Sigmoid] activation.
/// Not supported yet, because an optimization that combines the deriv of CCE and ReLU is hard
/// coded right now for optimization.
///
/// [Sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
struct SigmoidActivator;

impl Activator for SigmoidActivator {
    /// a(z) = 1 / 1 + e^-z
    fn activate(&self, z: f64) -> f64 {
        1.0 / (1.0 + -z.exp())
    }

    /// a' = a . (1 - a)
    fn derivative(&self, z: f64) -> f64 {
        let a = self.activate(z);
        a * (1.0 - a)
    }
}

/// A noop activator, used in the output layer sometimes when the activation/derivation is done
/// in a separate process (i.e combining activation + loss as an optimization).
struct NoopActivator;

impl Activator for NoopActivator {
    fn activate(&self, z: f64) -> f64 {
        z
    }

    fn derivative(&self, _: f64) -> f64 {
        1.0
    }
}

pub trait Loss {
    /// Computes the loss for a single sample.
    fn sample_loss(expected: &[f64], predicted: &[f64]) -> f64;

    /// Computes the mean loss for all samples in the batch.
    #[inline]
    fn mean_loss(sample_losses: &[f64]) -> f64 {
        Self::total_loss(sample_losses) / sample_losses.len() as f64
    }

    /// Computes the total loss for all samples in the batch.
    #[inline]
    fn total_loss(sample_losses: &[f64]) -> f64 {
        sample_losses.iter().sum()
    }
}

/// Categorical cross entropy implementation of a loss function.
///
/// ye: expected
/// yp: predicted
/// L = -Σye.ln(yp)
///
/// Could have been simplified to the following if instead of expected vec, we get the index of
/// the class that should be 1, because we know other classes will be 0.
/// L = -ln(yp)
pub struct CCELoss;

impl Loss for CCELoss {
    fn sample_loss(expected: &[f64], predicted: &[f64]) -> f64 {
        if expected.len() != predicted.len() {
            panic!("Mismatch of length.")
        }

        let predicted: Vec<_> = predicted
            .iter()
            .map(|v| v.clamp(EPSILON, 1.0 - EPSILON))
            .collect();

        let mut sum = 0.0;
        for i in 0..expected.len() {
            sum += expected[i] * predicted[i].ln();
        }

        -sum
    }
}

fn generate_random_neuron_params(count: usize) -> NeuronParams {
    NeuronParams {
        weights: generate_random_weights(count),
        bias: 0f64,
    }
}

fn generate_random_weights(count: usize) -> Vector {
    let mut weights = Vec::with_capacity(count);

    for _ in 0..count {
        let r = rnd_normal();
        weights.push(r * 0.01);
    }

    Vector::from_vec(weights)
}
