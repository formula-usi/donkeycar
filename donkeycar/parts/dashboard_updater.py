class DashboardUpdater:

    def __init__(self, socket_update_fn = None):
        self.socket_update_fn = socket_update_fn

    def run(self, steering, throttle):
        # seems to be a python bug that takes a single argument
        # return makes it into two element tuple with empty last element.
        angle_to_print = float(steering) if abs(float(steering)) > 0.05 else 0

        changes = {"throttle": throttle, "angle": angle_to_print}
        if self.socket_update_fn is not None:
            self.socket_update_fn(changes)
