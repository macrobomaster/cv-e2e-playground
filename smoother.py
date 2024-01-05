import math

class Smoother:
  def __init__(self, min_cutoff=0.7, beta=3.7, d_cutoff=1.0):
    self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
    self.x_prev, self.dx_prev, self.state = 0.0, 0.0, 0.0

  def update(self, measurement, dt):
    a_d = smoothing_factor(dt, self.d_cutoff)
    dx = (measurement - self.x_prev) / dt
    dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

    cutoff = self.min_cutoff + self.beta * abs(dx_hat)
    a = smoothing_factor(dt, cutoff)
    x_hat = exponential_smoothing(a, measurement, self.x_prev)

    self.x_prev, self.dx_prev, self.state = x_hat, dx_hat, x_hat
    return x_hat

def smoothing_factor(dt, cutoff):
  r = 2 * math.pi * cutoff * dt
  return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
  return a * x + (1 - a) * x_prev
