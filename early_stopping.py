class EarlyStopping:
    # early stop when <tolerance> occurances where loss does not decrease by at least <min_delta>

    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance # number of occurences to allow for
        self.min_delta = min_delta # minimum loss decrease 
        self.counter = 0
        self.early_stop = False
        self.prevLoss = None

    def __call__(self, loss):
        if self.prevLoss:
            if (self.prevLoss - loss) < self.min_delta:
                self.counter += 1
            
        self.prevLoss = loss

        if self.counter >= self.tolerance:  
            self.early_stop = True
            return True
        return False
