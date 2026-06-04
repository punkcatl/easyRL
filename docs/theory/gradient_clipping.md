# Gradient Clipping

## What is Gradient Clipping

During backpropagation, gradients can occasionally become very large (gradient explosion), causing model parameters to change drastically and destabilizing training. Gradient clipping limits the total gradient magnitude to prevent this.

## PyTorch API

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

- Must be called **after** `loss.backward()` and **before** `optimizer.step()`
- `max_norm`: the maximum allowed L2 norm of all gradients combined
- If the total norm exceeds `max_norm`, all gradients are scaled down proportionally
- Direction is preserved, only the step size is reduced

## How It Works

1. Collect all parameter gradients and treat them as one big vector
2. Compute the L2 norm (vector length): $\text{norm} = \sqrt{\sum g_i^2}$
3. If norm > max_norm, scale all gradients by factor `max_norm / norm`
4. Otherwise, do nothing

## Example

Suppose a network has 3 parameters. After `backward()`, their gradients are:

```
grad_1 = 12
grad_2 = 16
grad_3 = 0
```

**Step 1: Compute total norm**

```
norm = sqrt(12² + 16² + 0²) = sqrt(144 + 256) = sqrt(400) = 20
```

**Step 2: Compare with max_norm=10**

```
20 > 10 → need to clip
```

**Step 3: Scale all gradients by (max_norm / norm) = 10/20 = 0.5**

```
grad_1 = 12 * 0.5 = 6
grad_2 = 16 * 0.5 = 8
grad_3 = 0  * 0.5 = 0
```

**Verify:** `sqrt(6² + 8² + 0²) = sqrt(100) = 10 = max_norm`

The gradient direction stays the same (still going downhill in the same direction), only the magnitude is capped.

## Analogy

Going downhill: if the slope is too steep, you'd fall. Gradient clipping limits your maximum step size so you descend safely, without changing the direction you're heading.

## When to Use

- RL training (TD targets are noisy, can produce large errors)
- RNNs (long sequences cause gradient explosion through many time steps)
- Any training where occasional gradient spikes are observed

In simple environments like CartPole, gradient explosion is rare, so clipping is optional. But it costs almost nothing and adds robustness.

---

# 梯度裁剪

## 什么是梯度裁剪

反向传播时，梯度偶尔会变得非常大（梯度爆炸），导致模型参数剧烈变化，训练不稳定。梯度裁剪通过限制梯度的总大小来防止这种情况。

## PyTorch API

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

- 必须在 `loss.backward()` 之后、`optimizer.step()` 之前调用
- `max_norm`：所有梯度合在一起的最大 L2 范数
- 如果总范数超过 `max_norm`，所有梯度按比例等比缩小
- 方向不变，只是步长缩短

## 工作原理

1. 收集所有参数的梯度，当作一个大向量
2. 计算 L2 范数（向量长度）：$\text{norm} = \sqrt{\sum g_i^2}$
3. 如果 norm > max_norm，将所有梯度乘以缩放系数 `max_norm / norm`
4. 否则不做任何操作

## 举例

假设网络有 3 个参数，`backward()` 后各自的梯度为：

```
grad_1 = 12
grad_2 = 16
grad_3 = 0
```

**第 1 步：计算总范数**

```
norm = sqrt(12² + 16² + 0²) = sqrt(144 + 256) = sqrt(400) = 20
```

**第 2 步：和 max_norm=10 比较**

```
20 > 10 → 需要裁剪
```

**第 3 步：所有梯度乘以 (max_norm / norm) = 10/20 = 0.5**

```
grad_1 = 12 * 0.5 = 6
grad_2 = 16 * 0.5 = 8
grad_3 = 0  * 0.5 = 0
```

**验证：** `sqrt(6² + 8² + 0²) = sqrt(100) = 10 = max_norm`

梯度方向不变（还是朝同一个方向下山），只是大小被限制住了。

## 类比

下山时坡太陡会摔倒，梯度裁剪就是限制每步最大步长，保证安全下山，方向不变。

## 什么时候用

- RL 训练（TD 目标有噪声，可能产生大误差）
- RNN（长序列通过多个时间步导致梯度爆炸）
- 任何观察到偶发梯度尖峰的训练

在 CartPole 这样简单的环境中，梯度爆炸很少发生，裁剪是可选的。但它几乎没有额外开销，增加了鲁棒性。
