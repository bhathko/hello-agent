from abc import ABC, abstractmethod

class BaseExample(ABC):
    """
    Abstract base class for all example samples.
    """
    
    @abstractmethod
    def run(self):
        """
        Execute the sample logic for this example.
        """
        pass
