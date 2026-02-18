
class ModelTakeover(object):
    '''
    allow reverse to trigger automatic reverse throttle
    '''

    def __init__(self, angle_threshold=0.27, throttle_threshold=0.2, default_throttle=-0.2, n=50):
        self.angle_threshold = angle_threshold
        self.throttle_threshold = throttle_threshold
        self.default_throttle = default_throttle
        self.n = n
        self.angle_queue = [0.0] * n

    def run(self, angle, throttle, angle_unc, throttle_unc):
        # Filter out None values when calculating average
        valid_angles = [a for a in self.angle_queue if a is not None]
        avg_angle = sum(valid_angles) / len(valid_angles) if valid_angles else 0.0

        # If average angle is negative, set the multiplier to 1, otherwise set it to -1
        multiplier = 1 if avg_angle < 0 else -1
        if angle_unc is not None and throttle_unc is not None:
            if angle_unc > self.angle_threshold or throttle_unc > self.throttle_threshold:
                print("Model takeover")
                self.angle_queue = self.angle_queue[1:] + [multiplier]
                return multiplier, self.default_throttle
        # Only add angle to queue if it's not None, otherwise add 0.0
        self.angle_queue = self.angle_queue[1:] + [angle if angle is not None else 0.0]
        return angle, throttle
    
    def shutdown(self):
        pass
