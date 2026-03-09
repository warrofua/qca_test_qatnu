"""QATNU Simulator Textual frontend package."""

__all__ = ["QATNUSimulator"]


def __getattr__(name: str):
    if name == "QATNUSimulator":
        from .app import QATNUSimulator

        return QATNUSimulator
    raise AttributeError(name)
