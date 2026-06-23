from enum import IntEnum


    
class InteractionState(IntEnum):
    UNKNOWN = 0
    HELD_BY_HUMAN = 1
    THROWN_TO_ROBOT = 2
    IN_ROBOT_INTERACTION = 3
    THROWN_TO_HUMAN = 4
    FELL_TO_GROUND = 5
    IN_ROBOT_CONTACT = 6

