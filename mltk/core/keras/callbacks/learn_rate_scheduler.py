from typing import List,Tuple
import tensorflow as tf


from mltk.core import get_mltk_logger

class SteppedLearnRateScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        learning_rate:List[Tuple[int,float]]
    ):
        super().__init__()
        self.step = 0
        self.prev_lr = 0
        self.lr_schedule = [] 
        self.printed_schedule = False
        steps_sum = 0
        for step, lr in learning_rate:
            steps_sum += step
            self.lr_schedule.append((steps_sum, lr))


    def on_train_batch_begin(self, batch, logs=None):
        super().on_train_batch_begin(batch=batch, logs=logs)
        logger = get_mltk_logger()
        self.step += 1
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        self._print_schedule()

        for s,lr in self.lr_schedule:
            if self.step <= s:
                if self.prev_lr != lr:
                    self.prev_lr = lr
                    logger.info(f'\nStep {self.step}, updating learn rate: {lr}')
                tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))
                return

        logger.info(f"\n\n*** Maximum number of steps ({self.step}) exceeded. Stopping training\n")
        self.model.stop_training = True


    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch=batch, logs=logs)
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
        

    def _print_schedule(self):
        if not self.printed_schedule:
            logger = get_mltk_logger()
            self.printed_schedule = True 
            s = 'Learn rate schedule:\n'
            s += '  Less than step:\n'
            for step, lr in self.lr_schedule:
                s += f'  {step} --> {lr}\n'
            logger.info(s)

    