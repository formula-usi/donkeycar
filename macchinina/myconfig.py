# PWM_STEERING_THROTTLE
PWM_STEERING_THROTTLE = {
     "PWM_STEERING_PIN": "PCA9685.1:40.0",   # PWM output pin for steering servo
     "PWM_STEERING_SCALE": 1.0,              # to compensate for PWM frequency differences; NOT for adjusting steering range
     "PWM_STEERING_INVERTED": False,         # True if hardware requires an inverted PWM pulse
     "PWM_THROTTLE_PIN": "PCA9685.1:40.1",   # PWM output pin for ESC
     "PWM_THROTTLE_SCALE": 1.0,              # to compensate for PWM frequency differences; NOT for adjusting throttle
     "PWM_THROTTLE_INVERTED": False,         # True if hardware requires an inverted PWM pulse
     "STEERING_LEFT_PWM": 460,               # PWM value for full left steering
     "STEERING_RIGHT_PWM": 290,              # PWM value for full right steering
     "THROTTLE_FORWARD_PWM": 500,            # PWM value for max forward throttle
     "THROTTLE_STOPPED_PWM": 370,            # PWM value for no movement
     "THROTTLE_REVERSE_PWM": 220,            # PWM value for max reverse throttle
}

# CAMERA
CAMERA_VFLIP = True
CAMERA_HFLIP = True

# AI THROTTLE MULTIPLIER
AI_THROTTLE_MULT = 0.0                       # this multiplier will scale every throttle value for all output from NN models

# JOYSTICK
#USE_JOYSTICK_AS_DEFAULT = True              # When starting the manage.py it will not require a --js option to use the joystick
#JOYSTICK_MAX_THROTTLE = 0.35                # Scalar multiplied with the throttle value to limit the maximum throttle
#JOYSTICK_STEERING_SCALE = 1.0               # Scalar multiplied with the steering value to have a less sensitve steering