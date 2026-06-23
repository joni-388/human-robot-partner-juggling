"""
    mail@kaiploeger.net
"""

import os
import sys
from xmlrpc.client import ServerProxy

import InquirerPy as iq
import numpy as np
import rospy

from typing import Union, List


def select_from_topics_ending_with(end_of_topic_name: str, prompt_message: str='Select topic:') -> str:
    _, _, uri = rospy.get_master().getUri()
    master = ServerProxy(uri)
    _, _, all_topics = master.getTopicTypes(rospy.get_name())
    candidate_topic_names = [topic for topic in all_topics if topic[0].endswith(end_of_topic_name)]
    if not candidate_topic_names:
        raise RuntimeError(f"No topics found ending with '{end_of_topic_name}'.")
    return iq.prompt({'type': 'list',
                      'name': 'select_topic_ending_with',
                      'message': prompt_message,
                      'choices': candidate_topic_names},
                     vi_mode=True)["select_topic_ending_with"][0]


def select_from_topics(prompt: str='Select topic:',
                       starts_with: Union[str, None]=None,
                       contains: Union[str, List[str], None]=None,
                       ends_with: Union[str, None]=None) -> str:
    _, _, uri = rospy.get_master().getUri()
    master = ServerProxy(uri)
    _, _, candidate_topic_names = master.getTopicTypes(rospy.get_name())

    if starts_with is not None:
        candidate_topic_names = [topic for topic in candidate_topic_names if topic[0].startswith(starts_with)]

    if contains is not None:
        if isinstance(contains, str):
            contains_list = [contains]
        else:
            contains_list = list(contains)
        candidate_topic_names = [
            topic for topic in candidate_topic_names
            if any(substr in topic[0] for substr in contains_list)
        ]

    if ends_with is not None:
        candidate_topic_names = [topic for topic in candidate_topic_names if topic[0].endswith(ends_with)]

    if not candidate_topic_names:
        conditions = []
        if starts_with is not None:
            conditions.append(f"starts with '{starts_with}'")
        if contains is not None:
            if isinstance(contains, str):
                conditions.append(f"contains '{contains}'")
            else:
                conditions.append(f"contains {list(contains)}")
        if ends_with is not None:
            conditions.append(f"ends with '{ends_with}'")
        cond_str = ", ".join(conditions) if conditions else "no conditions"
        raise RuntimeError(f"No topics found matching the criteria: {cond_str}.")

    return iq.prompt({'type': 'list',
                      'name': 'selected_topic',
                      'message': prompt,
                      'choices': candidate_topic_names},
                     vi_mode=True)['selected_topic'][0]


def select_ros_param_containing(substring: str, prompt_message: str='Select parameter:') -> str:
    candidate_param_names = [param for param in rospy.get_param_names() if substring in param]
    if not candidate_param_names:
        raise RuntimeError(f"No parameter found containing '{substring}'.")
    return iq.prompt({'type': 'list',
                      'name': 'selected_param_containing',
                      'message': prompt_message,
                      'choices': candidate_param_names},
                      vi_mode=True)['selected_param_containing']


def select_ros_param(prompt: str='Select parameter:',
                     starts_with: Union[str,None]=None,
                     contains: Union[str,None]=None,
                     ends_with: Union[str,None]=None) -> str:
    candidate_param_names = rospy.get_param_names()

    if starts_with is not None: 
        candidate_param_names = [param for param in candidate_param_names if param[0].startswith(starts_with)]

    if contains is not None: 
        candidate_param_names = [param for param in candidate_param_names if contains in param[0]]

    if ends_with is not None: 
        candidate_param_names = [param for param in candidate_param_names if param[0].endswith(ends_with)]
 
    if not candidate_param_names:
        raise RuntimeError(f"No matching parameter found'.")

    return iq.prompt({'type': 'list',
                      'name': 'selected_param_containing',
                      'message': prompt_message,
                      'choices': candidate_param_names},
                      vi_mode=True)['selected_param_containing']


if os.name == 'posix':  # Unix/Linux
    import termios
    import tty

    def flush_input():
        """Flush the standard input buffer on Unix/Linux."""
        old_attrs = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)

else:  # Windows and other OS
    import msvcrt

    def flush_input():
        """Flush the standard input buffer on Windows."""
        while msvcrt.kbhit():
            msvcrt.getch()

