import math


class EarlyStopping:
    """Stop after patience consecutive scores without a strict improvement; zero disables."""
    def __init__(self, patience=10, min_delta=0.):
        if patience < 0 or min_delta < 0:
            raise ValueError("Patience and min_delta must be nonnegative")
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad_epochs = -math.inf, 0, 0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError("Validation score is not finite")
        improved = score > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.patience > 0 and self.bad_epochs >= self.patience

    def status(self, epochs_completed, stopped):
        return dict(epochs_completed=epochs_completed, best_epoch=self.best_epoch,
                    best_validation_dice=self.best, epochs_without_improvement=self.bad_epochs,
                    patience=self.patience, min_delta=self.min_delta,
                    stop_reason="validation_patience" if stopped else "maximum_epochs")
