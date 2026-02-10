from donkeycar.surface_handler import compute_throttle, compute_steering_angle


class WeatherApplier:
    def __init__(self, *args, **kwargs):
        self.surface = "dry"
        self.old_throttle = 0.0
        self.old_angle = 0.0




    def run(self, steering, throttle, surface):
        new_throttle = compute_throttle(throttle, self.old_throttle, surface)
        new_steering = compute_steering_angle(steering, throttle, self.old_angle, surface)
        return new_steering, new_throttle