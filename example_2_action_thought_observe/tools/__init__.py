"""
Tools package for the Hello-Agent application.
"""
from .weather import get_weather
from .attraction import get_attraction

__all__ = ['get_weather', 'get_attraction']


def get_available_tools() -> dict:
    """
    Get a dictionary of all available tools.

    Returns:
        dict: Tool name -> function mapping
    """
    return {
        'get_weather': get_weather,
        'get_attraction': get_attraction,
    }

