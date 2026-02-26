from abc import ABC, abstractmethod

class BaseChapter(ABC):
    """
    Abstract base class for all chapter samples.
    """
    
    @abstractmethod
    def run(self):
        """
        Execute the sample logic for this chapter.
        """
        pass
