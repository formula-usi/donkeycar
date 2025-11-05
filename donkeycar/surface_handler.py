import random
import math

def compute_throttle(throttle, old_throttle, surface):
    """
    Compute realistic throttle response based on surface conditions.
    
    Args:
        throttle: Current throttle input (-1 to 1)
        old_throttle: Previous throttle value
        surface: Surface type ("Dry", "Wet", "Frozen")
    """
    if surface == "Dry":
        return throttle
    elif surface == "Wet":
        return apply_wet_conditions(throttle, old_throttle)
    elif surface == "Frozen":
        return apply_icy_conditions(throttle, old_throttle)
    else:
        return throttle

def compute_steering_angle(steering_angle, old_steering_angle, surface):
    """
    Compute realistic steering response based on surface conditions.
    
    Args:
        steering_angle: Current steering input (-1 to 1)
        old_steering_angle: Previous steering value
        surface: Surface type ("Dry", "Wet", "Frozen")
    """
    if surface == "Dry":
        return steering_angle
    elif surface == "Wet":
        return apply_wet_steering(steering_angle, old_steering_angle)
    elif surface == "Frozen":
        return apply_icy_steering(steering_angle, old_steering_angle)
    else:
        return steering_angle

def apply_wet_conditions(throttle, old_throttle):
    """Simulate wet road conditions - reduced traction and wheel spin"""
    # Reduced acceleration on wet surfaces
    max_throttle_change = 0.7
    throttle_change = throttle - old_throttle
    
    # Limit sudden throttle changes to simulate wheel spin
    if abs(throttle_change) > 0.3:
        throttle_change = math.copysign(0.3, throttle_change)
    
    # Add slight unpredictability due to water patches
    noise = random.uniform(-0.05, 0.05)
    
    new_throttle = old_throttle + throttle_change * max_throttle_change + noise
    return max(-1, min(1, new_throttle))

def apply_icy_conditions(throttle, old_throttle):
    """Simulate icy road conditions - very limited traction and delayed response"""
    # Severely reduced acceleration/deceleration on ice
    max_throttle_change = 0.95
    throttle_change = throttle - old_throttle
    
    # Very limited throttle changes to simulate lack of traction
    if abs(throttle_change) > 0.15:
        throttle_change = math.copysign(0.15, throttle_change)
    
    # Add unpredictability and sliding effects
    if throttle_change > 0:
        noise = random.uniform(0, 0.2) if throttle != 0 else 0
    elif throttle_change < 0:
        noise = random.uniform(-0.2, 0) if throttle != 0 else 0
    # Simulate momentum - use throttle magnitude as proxy for "speed"
    momentum_factor = min(0.3, abs(old_throttle) * 0.5)
    
    new_throttle = old_throttle + throttle_change * max_throttle_change * (1 - momentum_factor) + noise
    return max(-1, min(1, new_throttle))

def apply_wet_steering(steering_angle, old_steering_angle):
    """Simulate wet road steering - reduced responsiveness and hydroplaning"""
    # Use throttle history to estimate if we're going fast (more aggressive steering = higher speed assumption)
    speed_proxy = abs(old_steering_angle) * 0.5  # Previous steering as speed indicator
    speed_factor = max(0.5, 1.0 - speed_proxy)
    
    # Limit sudden steering changes
    max_steering_change = 0.6 * speed_factor
    steering_change = steering_angle - old_steering_angle
    
    if abs(steering_change) > 0.4:
        steering_change = math.copysign(0.4, steering_change)
    
    # Add slight drift/understeer
    drift = random.uniform(-0.03, 0.03)
    
    new_steering = old_steering_angle + steering_change * max_steering_change + drift
    return max(-1, min(1, new_steering))

def apply_icy_steering(steering_angle, old_steering_angle):
    """Simulate icy road steering - oversteer, delayed response, unpredictable behavior"""
    # Use previous steering magnitude as proxy for difficulty
    difficulty_factor = max(0.3, 1.0 - abs(old_steering_angle) * 0.4)
    
    steering_change = steering_angle - old_steering_angle
    
    # On ice, small steering inputs can cause big changes (oversteer effect)
    if abs(steering_change) > 0.1:
        # Amplify steering on ice (oversteer effect)
        amplified_change = steering_change * random.uniform(1.5, 2.5)
        steering_change = math.copysign(min(abs(amplified_change), 0.8), steering_change)
    
    # Add significant unpredictability - ice patches, sliding
    slide_noise = random.uniform(-0.3, 0.3) 
    
    # Simulate delayed response
    response_delay = 0.7 * difficulty_factor
    
    new_steering = old_steering_angle + steering_change * response_delay + slide_noise
    
    # Ice can cause sudden slides when steering aggressively
    if abs(steering_change) > 0.3 and random.random() < 0.15:
        # Sudden slide/spin
        new_steering += random.uniform(-0.3, 0.3)
    
    return max(-1, min(1, new_steering))

# Legacy function for backward compatibility
def slow_break(throttle, old_throttle, preserved_speed_percentage):
    """Legacy function - preserved for backward compatibility"""
    if throttle < old_throttle:
        return throttle + (old_throttle - throttle) * preserved_speed_percentage
    return throttle