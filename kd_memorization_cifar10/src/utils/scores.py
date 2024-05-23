
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map
import jax as jax


def get_lord_error_fn(fn, params, state, ord):
  @jit
  def lord_error(X, Y):
    # errors = nn.softmax(fn(params, state, X)) - Y
    errors = nn.softmax(fn(params, X)) - Y
    scores = jnp.linalg.norm(errors, ord=ord, axis=-1)
    return scores
  np_lord_error = lambda X, Y: np.array(lord_error(X, Y))
  return np_lord_error


def get_margin_error(fn, params, state, score_type):
  fn_jit = jit(lambda X: fn(params, state, X))

  def margin_error(X, Y):
    batch_sz = X.shape[0]
    P = np.array(nn.softmax(fn_jit(X)))
    correct_logits = Y.astype(bool)
    margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
    if score_type == 'max':
      scores = np.max(margins, -1)
    elif score_type == 'sum':
      scores = np.sum(margins, -1)
    return scores

  return margin_error

def cross_entropy_loss(logits, labels):
  return jnp.mean(-jnp.sum(nn.log_softmax(logits) * labels, axis=-1))

def get_grad_norm_fn(fn, params, state):

  @jit
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(cross_entropy_loss)(fn(p, state, x), y)
    loss_grads = jnp.concatenate(tree_flatten(tree_map(vmap(jnp.ravel), (jacrev(per_sample_loss_fn)(params, X, Y))))[0], axis=1)
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_score_fn(fn, params, state, score_type):
  if score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 2)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 1)
  elif score_type == 'max_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'max')
  elif score_type == 'sum_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'sum')
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = get_grad_norm_fn(fn, params, state)
  else:
    raise NotImplementedError
  return score_fn


def compute_scores(fn, params, state, X, Y, batch_sz, score_type):
  n_batches = X.shape[0] // batch_sz
  Xs, Ys = np.split(X, n_batches), np.split(Y, n_batches)
  score_fn = get_score_fn(fn, params, state, score_type)
  scores = []
  for i, (X, Y) in enumerate(zip(Xs, Ys)):
    print(f'score batch {i+1} of {n_batches}')
    scores.append(score_fn(X, Y))
  scores = np.concatenate(scores)
  return scores


def compute_l2_error(model, params, state, X, Y, batch_size):
    return compute_scores(model, params, state, X, Y, batch_size, 'l2_error')

def compute_grad_norm(model, params, state, X, Y, batch_size):
    return compute_scores(model, params, state, X, Y, batch_size, 'grad_norm')



# # Define a simple neural network model
# class SimpleNN(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(features=10)(x)  # Output 10 logits for 10 classes
#         return x

# # Initialize model and parameters
# key = jax.random.PRNGKey(0)
# x_sample = jnp.ones((1, 5))  # Example input to determine parameter shapes
# model = SimpleNN()
# params = model.init(key, x_sample)

# # Generate synthetic data
# np.random.seed(0)
# num_samples = 100
# input_dim = 5
# num_classes = 10
# X = jnp.array(np.random.randn(num_samples, input_dim))
# Y = jnp.eye(num_classes)[np.random.choice(num_classes, num_samples)]  # One-hot encoded labels

# # Define a dummy state if your model requires it (usually for stateful layers)
# state = {}

# # Compute scores
# batch_sz = 10
# scores = compute_scores(model.apply, params, state, X, Y, batch_sz, 'l2_error')
# print("Computed l2_error scores:", scores)