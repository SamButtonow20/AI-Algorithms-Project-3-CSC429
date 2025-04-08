# MNIST Optimization Report

## Group Members
- Rizik Haddad
- Sam Buttonow

---

## Project Overview

The goal of this project was to build a binary classifier using the MNIST dataset and compare different optimization algorithms. We focused on training a logistic regression model to distinguish between the digit "0" and all other digits using the following optimization methods:

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- Adam (with and without bias correction)

Each method was implemented manually using NumPy to better understand how they work under the hood. No built-in PyTorch optimizers were used.

---

## Task 1 – SGD (Rizik)

Implemented and analyzed the behavior of vanilla SGD with three fixed learning rates: `0.1`, `0.01`, and `0.001`. Additionally, a version using a decreasing learning rate was included, based on the formula:

$$
\eta_t = \frac{\eta_0}{1 + k \cdot t}
$$

Where:
- `η₀ = 0.1`
- `k = 0.01`
- `t` is the iteration number

For each setting, I tracked the training loss and plotted how it changed over time to understand how the learning rate affects convergence.

---

## Task 2 – SGD with Momentum (Sam)

Implemented SGD with momentum to enhance convergence by incorporating a velocity term that accumulates past gradients. The update rule used was:

$$
\begin{aligned}
v &= \beta \cdot v + (1 - \beta) \cdot \nabla L(\theta) \\
\theta &= \theta - \alpha \cdot v
\end{aligned}
$$


Where:

- β = 0.9 is the momentum coefficient
- ∇L is the loss gradient
- α is the learning rate

Momentum was applied to both weights and bias. Performance was evaluated and compared against vanilla SGD.

---

## Task 3 – Adam Optimizer (Sam)

Implemented the Adam optimizer in two variants:

1. With bias correction (standard Adam)  
2. Without bias correction (for comparison)

Adam combines momentum and adaptive learning rates using the following update formulas:

$$
\begin{aligned}
m_t &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla L(\theta) \\
v_t &= \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla L(\theta))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta &= \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Bias correction ensures that the moment estimates are unbiased during early iterations, especially when \( t \) is small.

Loss was tracked during training for both versions and compared against SGD and Momentum.

---

## Hyperparameter Tuning Discussion


For SGD, three learning rates were tested (0.1, 0.01, 0.001) to evaluate convergence speed and stability. A decreasing learning rate variant provided a balanced approach. 

Momentum used β = 0.9, a standard value known to accelerate convergence while suppressing oscillations. 

For Adam, the default values β₁ = 0.9 and β₂ = 0.999 were used, as they generally work well in practice. 

The ε value was set to 1e-8 to avoid division by zero in parameter updates.

---

## Results and Observations

### SGD and Momentum Comparison

The following plot compares the performance of plain SGD with different learning rates and SGD with momentum.

![original image](https://cdn.mathpix.com/snip/images/wOs-9FMogN5lcmlRY7ARlsLYL-M5eBAeZvnxZ47TOXM.original.fullsize.png)


- **SGD (lr = 0.1)**: Fast initial convergence but fluctuated a lot.
- **SGD (lr = 0.01)**: Balanced speed and stability, and converged well.
- **SGD (lr = 0.001)**: Very slow, not practical unless extremely stable learning is required.
- **SGD (decreasing lr)**: Started strong like lr=0.1 but smoothed out over time — good overall behavior.
- **Momentum**: More stable and faster than all SGD variants.

---

### Adam Comparison

This plot compares the Adam optimizer with and without bias correction.

![original image](https://cdn.mathpix.com/snip/images/xZvwIG-FkmFPh7J1O0Opp9hnld6GMejqrNbzMTa8lt4.original.fullsize.png)


- **With bias correction**: Most stable and fastest convergence overall.
- **Without bias correction**: Still effective, but less stable early on.

---

## Final Loss Values

```bash
Final SGD lr=0.1 Loss: 0.0027
Final SGD lr=0.01 Loss: 0.0219
Final SGD lr=0.001 Loss: 0.1479
Final SGD (decreasing lr) Loss: 0.0279
Final Momentum Loss: 0.0358
Final Adam (bias corrected) Loss: 0.0180
Final Adam (no bias) Loss: 0.0003
```

---

## Summary

Learning rate strongly impacts convergence in SGD. High rates converge quickly but risk instability. Low rates are stable but slow. A decreasing learning rate offers a good compromise. Momentum improves convergence speed and smoothness. Adam achieves the best performance overall, especially when bias correction is used.

---

## How to Run the Code

1. Install dependencies:
    ```
    pip install torch torchvision matplotlib numpy
    ```

2. Run the script:
    ```
    python main.py
    ```

3. Output plots will be saved to the `plots/` folder, and loss values will be printed in the terminal.

---
