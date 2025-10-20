class CustomEarlyStopper:
    def __init__(self, patience=5, min_delta=0.001, mode="min"):
        """
        Early stopping utility.
        Args:
            patience (int): Number of validations with no improvement before stop.
            min_delta (float): Minimum change to count as improvement.
            mode (str): "min" for metrics to minimize (e.g. loss), "max" for maximize (e.g. accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        acceptable_modes = ['min', 'max']
        if mode not in acceptable_modes:
            raise ValueError("Incorrect arg provided for early stopper mode. Acceptable values: {}".format(acceptable_modes))
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, metric):
        # Determine if there is an improvement
        if self.best is None:
            self.best = metric
            self.bad_epochs = 0
            return False
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        elif self.mode == 'max':
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience  # True means should stop