from nnimputer import NNImputer
import numpy


class WassersteinNN(NNImputer):
    def __init__(self):
        return

    def cross_validate(self, *args, **kwargs):
        return super().cross_validate(*args, **kwargs)

    def estimate(self, *args, **kwargs):
        return super().estimate(*args, **kwargs)

    def distances(self, *args, **kwargs):
        return super().distances(*args, **kwargs)
