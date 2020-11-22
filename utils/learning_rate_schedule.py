import tensorflow as tf
import numpy as np

"""
Custom learning rate scheduler for yolact 
"""

# https://github.com/tensorflow/models/blob/1b5a4c9ed33242783eaf29e664618331dbb59e1b/research/object_detection/utils/learning_schedules.py#L85
class Yolact_LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    """

    def __init__(self, warmup_steps, warmup_lr, initial_lr, total_steps):
        """
        :param warmup_steps:
        :param warmup_lr:
        :param initial_lr:
        """
        super(Yolact_LearningRateSchedule, self).__init__()
        self.warmup_step = warmup_steps
        self.warmup_lr = warmup_lr
        self.initial_lr = initial_lr
        self.total_steps = total_steps

    def __call__(self, global_step, hold_base_rate_steps=0):
        """
        Args:
        global_step: int64 (scalar) tensor representing global step.
        hold_base_rate_steps: Optional number of steps to hold base learning rate
          before decaying.

        Returns:
        If executing eagerly:
          returns a no-arg callable that outputs the (scalar)
          float tensor learning rate given the current value of global_step.
        If in a graph:
          immediately returns a (scalar) float tensor representing learning rate.

        Raises:
        ValueError: if warmup_learning_rate is larger than learning_rate_base,
          or if warmup_steps is larger than total_steps.
        """
        warmup_learning_rate = tf.convert_to_tensor(self.warmup_lr)
        dtype = warmup_learning_rate.dtype
        warmup_steps = tf.cast(self.warmup_step, dtype)
        learning_rate_base = tf.cast(self.initial_lr, dtype)
        learning_rate_base = tf.cast(self.initial_lr, dtype)
        total_steps = tf.cast(self.total_steps, dtype)

        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
            np.pi *
            (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
            ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
          learning_rate = tf.where(
              global_step > warmup_steps + hold_base_rate_steps,
              learning_rate, learning_rate_base)
        if warmup_steps > 0:
          if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
          slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
          warmup_rate = slope * tf.cast(global_step,
                                        tf.float32) + warmup_learning_rate
          learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                   learning_rate)
        return tf.where(global_step > total_steps, 0.0, learning_rate,
                        name='learning_rate')

    def get_config(self):
        return {
            "warm up learning rate": self.warmup_lr,
            "warm up steps": self.warmup_steps,
            "initial learning rate": self.initial_lr
        }
