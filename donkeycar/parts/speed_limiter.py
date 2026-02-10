class SpeedLimiter:

    def __init__(self):
        self.max_throttle = 1.0
        self.straight_throttle = 1.0
        self.steer_throttle = 1.0
        self.throttle_mode = "max"
        self.old_throttle = 0.0
        self.old_angle = 0.0

    def limited_throttle(
        self,
        new_throttle,
        max_throttle,
        throttle_mode,
        straight_throttle,
        steer_throttle,
        angle
    ):
        limited_throttle = 0

        if new_throttle > 0:
            limited_throttle = min(max_throttle, new_throttle)

        if new_throttle < 0:
            limited_throttle = max(-max_throttle, new_throttle)

        if throttle_mode == "constant":
            limited_throttle = max_throttle

        if throttle_mode == "steer_limited":
            # Interpolate between straight throttle and full steer throttle
            steer_amount = abs(angle)  # 0 to 1
            max_allowed_throttle = (
                straight_throttle +
                (steer_throttle - straight_throttle) * steer_amount
            )

            if new_throttle > 0:
                limited_throttle = min(max_allowed_throttle, new_throttle)
            elif new_throttle < 0:
                limited_throttle = max(-max_allowed_throttle, new_throttle)

        return limited_throttle


    def run(self, steering, throttle):
        # seems to be a python bug that takes a single argument
        # return makes it into two element tuple with empty last element.
        throttle = self.limited_throttle(
            throttle,
            self.max_throttle,
            self.throttle_mode,
            self.straight_throttle,
            self.steer_throttle,
            steering
        )
        return steering, throttle
