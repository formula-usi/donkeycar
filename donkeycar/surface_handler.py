import random
import math

def compute_throttle(throttle, old_throttle, surface):
    """
    Compute realistic throttle response based on surface conditions.
    
    Args:
        throttle: Current throttle input (-1 to 1)
        old_throttle: Previous throttle value
        surface: Surface type ("Dry", "Wet", "Icy")
    """
    throttle = throttle if abs(throttle) > 0.1 else 0

    if surface.upper() == "DRY":
        return throttle
    elif surface.upper() == "WET":
        return apply_wet_conditions(throttle, old_throttle)
    elif surface.upper() == "ICY":
        return apply_icy_conditions(throttle, old_throttle)
    else:
        return throttle

def compute_steering_angle(steering_angle, throttle, old_steering_angle, surface):
    """
    Compute realistic steering response based on surface conditions.
    
    Args:
        steering_angle: Current steering input (-1 to 1)
        throttle: Current throttle input (-1 to 1)
        old_steering_angle: Previous steering value
        surface: Surface type ("Dry", "Wet", "Icy")
    """
    throttle = throttle if abs(throttle) > 0.1 else 0
    if surface.upper() == "DRY":
        return steering_angle
    elif surface.upper() == "WET":
        return apply_wet_steering(steering_angle, throttle, old_steering_angle)
    elif surface.upper() == "ICY":
        return apply_icy_steering(steering_angle, throttle, old_steering_angle)
    else:
        return steering_angle

def apply_wet_conditions(throttle, old_throttle):
    """Simulate wet road conditions - reduced traction and wheel spin"""
    # Reduced acceleration on wet surfaces
    max_throttle_change = 0.85
    noise = 0.6
    throttle_inversion_probability = 0
    max_throttle_inversion = 0
    return apply_throttle_conditions(throttle, old_throttle, max_throttle_change, noise, throttle_inversion_probability, max_throttle_inversion)

def apply_icy_conditions(throttle, old_throttle):
    """Simulate icy road conditions - very limited traction and delayed response"""
    # Severely reduced acceleration/deceleration on ice
    max_throttle_change = 0.95
    noise = 0.9
    throttle_inversion_probability = 0.1
    max_throttle_inversion = 0.9
    return apply_throttle_conditions(throttle, old_throttle, max_throttle_change, noise, throttle_inversion_probability, max_throttle_inversion)
    
def throttle_inversion(throttle, max_inversion):
    if throttle > 0:
        return throttle - max_inversion * throttle
    elif throttle < 0:
        return throttle + max_inversion * abs(throttle)
    return throttle

def apply_throttle_conditions(throttle, old_throttle, max_throttle_change, max_noise, throttle_inversion_probability=0, max_throttle_inversion=0):
    throttle_change = throttle - old_throttle
    accellerating = False
    if throttle_change >= 0 and throttle > 0:
        accellerating = True  # Reduce acceleration on slippery surfaces
    elif throttle_change <= 0 and throttle < 0:
        accellerating = True  # Reduce braking on slippery surfaces
    
    # # Very limited throttle changes to simulate lack of traction
    # if abs(throttle_change) > 0.15:
    #     throttle_change = math.copysign(0.15, throttle_change)
    
    # Add unpredictability and sliding effects
    if throttle_change >= 0:
        noise = random.uniform(0, max_noise)*throttle if throttle != 0 else 0
    else:
        noise = random.uniform(-max_noise, 0)*throttle if throttle != 0 else 0
    
    # Simulate momentum - use throttle magnitude as proxy for "speed"

    if accellerating == False:
        momentum_factor = min(0.3, abs(old_throttle) * 0.5)
        multiplier = max_throttle_change * (1 - momentum_factor)
    else:
        multiplier = 1

    
    new_throttle = old_throttle + throttle_change * multiplier + noise
    throttle_not_inverted =  max(-1, min(1, new_throttle))

    #random throttle inversion to simulate loss of control
    if throttle_inversion_probability > 0 and random.random() < throttle_inversion_probability:
        return throttle_inversion(throttle_not_inverted, max_throttle_inversion)
    return throttle_not_inverted

def apply_wet_steering(steering_angle, throttle,  old_steering_angle):
    """Simulate wet road steering - reduced responsiveness and hydroplaning"""
    delay = 0.6
    noise = 0.2
    probability_aggressive_noise = 0.05

    return apply_steering(steering_angle, throttle, old_steering_angle, delay, noise, probability_aggressive_noise)

def apply_icy_steering(steering_angle, throttle, old_steering_angle):
    delay = 0.7
    noise = 0.25
    probability_aggressive_noise = 0.3
    return apply_steering(steering_angle, throttle, old_steering_angle, delay, noise, probability_aggressive_noise)
    

def apply_steering(steering_angle, throttle, old_steering_angle, delay, noise, probability_aggressive_noise):
    """Simulate icy road steering - oversteer, delayed response, unpredictable behavior"""
    # Use previous steering magnitude as proxy for difficulty
    difficulty_factor = max(0.3,  1 - abs(old_steering_angle) * 0.4)
    
    steering_change = steering_angle - old_steering_angle
    # print(f"CHANGE {steering_change} , OLD {old_steering_angle} , NEW {steering_angle}")
    # On ice, small steering inputs can cause big changes (oversteer effect)
    if abs(steering_change) > 0.1:
        # Amplify steering on ice (oversteer effect)
        amplified_change = steering_change * random.uniform(1.5, 2.5)
        steering_change = math.copysign(min(abs(amplified_change), 0.8), steering_change)
    
    # Add significant unpredictability - ice patches, sliding
    slide_noise = random.uniform(-noise, noise) if throttle != 0 else 0
    # slide_noise = 0
    # Simulate delayed response
    response_delay = delay * difficulty_factor
    # print(old_steering_angle, steering_change, slide_noise)
    # response_delay = 1
    
    new_steering = old_steering_angle + steering_change * response_delay + slide_noise
    # print(f"SLIDE NOISE {slide_noise}")
    
    # Ice can cause sudden slides when steering aggressively
    if abs(steering_change) > 0.3 and random.random() < probability_aggressive_noise and throttle != 0:
        # Sudden slide/spin
        new_steering += random.uniform(-0.25, 0.25)
    
    return max(-1, min(1, new_steering))

# Legacy function for backward compatibility
def slow_break(throttle, old_throttle, preserved_speed_percentage):
    """Legacy function - preserved for backward compatibility"""
    if throttle < old_throttle:
        return throttle + (old_throttle - throttle) * preserved_speed_percentage
    return throttle