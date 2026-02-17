
class DirectionEstimator(object):
    '''
    allow reverse to trigger automatic reverse throttle
    '''

    def __init__(self, n=10):
        #create a queue with n zeros
        self.n = n
        self.angle_queue = [0.0] * n

    def run(self, angle_in):
        if angle_in is None:
            return angle_in

        # Update the queue with the new angle value
        self.angle_queue = self.angle_queue[1:] + [angle_in]

        # Calculate average angle over the last n values
        avg_angle = sum(self.angle_queue) / len(self.angle_queue)

        # If average angle is negative, set the multiplier to -1, otherwise set it to 1
        multiplier = -1 if avg_angle < 0 else 1
        return multiplier

    def shutdown(self):
        pass
