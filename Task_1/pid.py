from enum import Flag, auto

from numpy import float64


class PIDComponents(Flag):
    PROPORTIONAL = auto()
    INTEGRAL = auto()
    DERIVATIVE = auto()
    PI = PROPORTIONAL | INTEGRAL
    PID = PROPORTIONAL | INTEGRAL | DERIVATIVE

class PID:
    DEFAULT_PROPORTIONAL_COEFF: float64 = 1
    DEFAULT_INTEGRAL_COEFF: float64 = 1
    DEFAULT_DERIVATIVE_COEFF: float64 = 1

    proportional_coeff: float64 = DEFAULT_PROPORTIONAL_COEFF
    integral_coeff: float64 = DEFAULT_INTEGRAL_COEFF
    derivative_coeff: float64 = DEFAULT_DERIVATIVE_COEFF
    
    _step_time: float64
    _components: PIDComponents
    _position_diff: float64 = 0
    _previous_position_diff: float64 = 0
    _cumulative_position_diff: float64 = 0

    def __init__(self, step_time: float64, components: PIDComponents):
        self._step_time = step_time
        self._components = components

    def reset(self):
        self._previous_position_diff = 0
        self._cumulative_position_diff = 0

    def get_control(self, target_position: float64, current_position: float64) -> float64:
        self._previous_position_diff = self._position_diff
        self._position_diff = target_position - current_position

        result: float64 = self._get_component(PIDComponents.PROPORTIONAL) + \
                          self._get_component(PIDComponents.INTEGRAL) + \
                          self._get_component(PIDComponents.DERIVATIVE)
        
        self._cumulative_position_diff += self._position_diff
        return result
    
    def _get_component(self, component: PIDComponents) -> float64:
        if component not in self._components:
            return 0

        match component:
            case PIDComponents.PROPORTIONAL:
                return self.proportional_coeff * self._position_diff
            case PIDComponents.INTEGRAL:
                return self.integral_coeff * self._cumulative_position_diff * self._step_time
            case PIDComponents.DERIVATIVE:
                return self.derivative_coeff * (self._position_diff - self._previous_position_diff) / self._step_time
            
    