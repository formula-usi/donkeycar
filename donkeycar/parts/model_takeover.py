
class ModelTakeover(object):
    '''
    allow reverse to trigger automatic reverse throttle
    '''

    def __init__(self, angle_threshold=0.27, throttle_threshold=0.2, default_throttle=-0.2):
        self.angle_threshold = angle_threshold
        self.throttle_threshold = throttle_threshold
        self.default_throttle = default_throttle

    def run(self, angle, throttle, angle_unc, throttle_unc, multiplier):
        if angle_unc is not None and throttle_unc is not None:
            if angle_unc > self.angle_threshold or throttle_unc > self.throttle_threshold:
                return multiplier, self.default_throttle
        return angle, throttle
    def shutdown(self):
        pass
