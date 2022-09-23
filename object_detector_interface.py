import abc


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def load_class_name(self):
        pass

    @abc.abstractmethod
    def detect_object(self):
        pass
