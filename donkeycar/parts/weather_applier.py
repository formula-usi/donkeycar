from donkeycar.surface_handler import compute_throttle, compute_steering_angle


class WeatherApplier:
    def __init__(self, *args, **kwargs):
        self.surface = "dry"
        self.old_throttle = 0.0
        self.old_angle = 0.0
        self.dead_steer = 0.05




    def run(self, steering, throttle, surface):
        if abs(steering) < self.dead_steer:
            steering = 0.0
        new_throttle = compute_throttle(throttle, self.old_throttle, surface)
        new_steering = compute_steering_angle(steering, throttle, self.old_angle, surface)
        self.old_throttle = throttle
        self.old_angle = steering


        return new_steering, new_throttle