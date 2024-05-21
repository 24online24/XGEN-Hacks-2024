from dataclasses import dataclass


@dataclass
class News:
    title: str
    content: str


@dataclass
class ResponseArrayElement:
    name: str
    description: str
    value: float
