import time

from typing import Any, Dict, Tuple, List, ClassVar, Callable, Optional
from dataclasses import dataclass, field
from contextlib import ContextDecorator

from psutil import virtual_memory

@dataclass
class Timer(ContextDecorator):
    """
    Implemented a timer class following the tutorial from https://realpython.com/python-timer/#using-the-python-timer-context-manager
    """
    timers : ClassVar[Dict[str, float]] = dict()
    verbose : ClassVar[int] = 0

    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f}s"
    mode : int = 0
    log : Optional[Callable[[str], None]] = print
    startTime : Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name:
            self.timers.setdefault(self.name,0)

    def start(self) -> None:
        self.startTime = time.perf_counter()

    def stop(self) -> float:
        elapsedTime = time.perf_counter() - self.startTime
        self.startTime = None

        if self.log and self.mode >= self.verbose:
            self.log(self.text.format(elapsedTime))

        if self.name:
            self.timers[self.name] += elapsedTime

        return elapsedTime

    #for the context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    """#for the decorator
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_time(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper_time()
    """

@dataclass
class Memory(ContextDecorator):
    watcher : ClassVar[Dict[str, Tuple[float, float, float]]] = dict()
    verbose : ClassVar[int] = 0

    name: Optional[str] = None
    pre_text : str = ""
    text: str = "Total: {tot}GB, d(Available): {ava}GB, d(Free): {fre}GB, d(Used): {use}GB"
    mode : int = 0
    log : Optional[Callable[[str], None]] = print
    startMemory : Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name:
            self.watcher.setdefault(self.name,(0.0,0.0,0.0))

    def start(self) -> None:
        self.startMemory = virtual_memory()

    def stop(self) -> float:
        current_memory = virtual_memory()
        dif_available = (current_memory.available - self.startMemory.available) / 1024**3
        dif_free = (current_memory.free - self.startMemory.free) / 1024**3
        dif_used = (current_memory.used - self.startMemory.used) / 1024**3

        self.startMemory = None

        if self.log and self.mode >= self.verbose:
            self.log(self.pre_text + self.text.format(tot=current_memory.total, ava=dif_available, fre=dif_free, use=dif_used))

        if self.name:
            self.watcher[self.name][0] += dif_available
            self.watcher[self.name][1] += dif_free
            self.watcher[self.name][2] += dif_used


        return dif_available

    #for the context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
