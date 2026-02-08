# Overview
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=E4f8bKYJ-Bd1nO6u)

```bash
conda create -n gpt-env -y python=3.14
conda activate gpt-env
```

## Backpropagation
Backpropagation computes gradients by applying the chain rule backwards through a computational graph. Each operation has a specific effect on derivatives:
### Operations and Their Derivatives
**Addition (+)**: `z = x + y`
- `∂z/∂x = 1` and `∂z/∂y = 1`
- Gradient flows equally to both operands
**Multiplication (*)**: `z = x * y`  
- `∂z/∂x = y` and `∂z/∂y = x`
- Gradient is scaled by the other operand's value
**Tanh**: `z = tanh(x)`
- `∂z/∂x = 1 - tanh²(x)`
- Gradient is scaled by the derivative of the activation function
- **Role**: Non-linear activation that introduces non-linearity and squashes outputs to [-1, 1], preventing exploding gradients while learning complex patterns
### Chain Rule
When operations are composed (e.g., `L = tanh(w*x + b)`), gradients multiply through each step:
- `∂L/∂w = ∂L/∂tanh * ∂tanh/∂(w*x+b) * ∂(w*x+b)/∂w`
- This allows computing how each parameter affects the final output
### Gradient Accumulation
Gradients accumulate when a node is used multiple times in the graph, ensuring all paths contribute to the final derivative.

## Neural Network Implementation

We implemented a complete neural network from scratch using our custom `Value` class that supports automatic differentiation:

### Core Components
- **Value Class**: Custom scalar type that tracks operations and enables backpropagation through computational graphs
- **Automatic Differentiation**: Implements gradient computation for `+`, `*`, `tanh`, `exp`, and power operations
- **Backpropagation**: Uses topological sorting to efficiently compute gradients backwards through the network
- **Neuron**: Single unit with weights, bias, and tanh activation
- **Multi-Layer Perceptron**: Stacked layers of neurons for complex function approximation

### Training Process
1. **Forward Pass**: Compute predictions by passing inputs through the network
2. **Loss Calculation**: Mean squared error between predictions and targets using `sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred), Value(0.0))`
3. **Backward Pass**: Compute gradients of loss with respect to all parameters
4. **Parameter Updates**: Adjust weights and biases using gradient descent

### Key Features
- **Visual Computational Graphs**: Used Graphviz to visualize forward/backward passes and gradient flow
- **Gradient Accumulation**: Properly handles cases where variables are reused multiple times
- **Non-linearity**: Tanh activation enables learning complex, non-linear relationships

**What is a Neural Network?** A neural network is a computational model inspired by biological neurons that learns to approximate complex functions by adjusting parameters (weights and biases) through gradient-based optimization. It transforms inputs through multiple layers of non-linear transformations to produce desired outputs.

### Common Training Errors
2. **Gradient Accumulation Bug**: Forgetting to zero gradients between training batches causes incorrect gradient accumulation, leading to unstable training. **Solution**: Reset all parameter gradients to 0 before each backward pass.

This implementation demonstrates the fundamental mechanics of modern deep learning frameworks, building from mathematical first principles to a working neural network that can learn from data.

## Example: Manual Backpropagation for L

Looking at the example `L = d * f` where `d = e + c` and `e = a * b`, let's manually walk through the backpropagation:

### Forward Pass
```
a = 2.0, b = -3.0, c = 10.0, f = -2.0
e = a * b = 2.0 * -3.0 = -6.0
d = e + c = -6.0 + 10.0 = 4.0  
L = d * f = 4.0 * -2.0 = -8.0
```

### Backward Pass (Chain Rule)

**Step 1: Initialize**
```
∂L/∂L = 1.0  (base case)
L.grad = 1.0
```

**Step 2: L = d * f**
For multiplication: `∂(x*y)/∂x = y` and `∂(x*y)/∂y = x`
```
∂L/∂d = f = -2.0
∂L/∂f = d = 4.0

d.grad = -2.0
f.grad = 4.0
```

**Step 3: d = e + c**  
For addition: `∂(x+y)/∂x = 1` and `∂(x+y)/∂y = 1`
```
∂L/∂e = ∂L/∂d * ∂d/∂e = -2.0 * 1 = -2.0
∂L/∂c = ∂L/∂d * ∂d/∂c = -2.0 * 1 = -2.0

e.grad = -2.0
c.grad = -2.0
```

**Step 4: e = a * b**
For multiplication: `∂(x*y)/∂x = y` and `∂(x*y)/∂y = x`
```
∂L/∂a = ∂L/∂e * ∂e/∂a = -2.0 * b = -2.0 * -3.0 = 6.0
∂L/∂b = ∂L/∂e * ∂e/∂b = -2.0 * a = -2.0 * 2.0 = -4.0

a.grad = 6.0
b.grad = -4.0
```

### Final Gradients
```
a.grad = 6.0   (∂L/∂a)
b.grad = -4.0  (∂L/∂b) 
c.grad = -2.0  (∂L/∂c)
e.grad = -2.0  (∂L/∂e)
d.grad = -2.0  (∂L/∂d)
f.grad = 4.0   (∂L/∂f)
L.grad = 1.0   (∂L/∂L)
```

### Interpretation
- Increasing `a` by 0.01 would increase `L` by ~0.06
- Increasing `b` by 0.01 would decrease `L` by ~0.04  
- Increasing `c` by 0.01 would decrease `L` by ~0.02
- Increasing `f` by 0.01 would increase `L` by ~0.04

This manual walkthrough shows how the chain rule propagates gradients backward through each operation, computing how each input parameter affects the final output.

### Common Errors
1. If you reuse multiple variables, for example:
```text
a = Value(3.0)
b = a + a
```
then the gradients are going to be inaccurate.

**to solve this, the gradients must be added together**
2. For training 