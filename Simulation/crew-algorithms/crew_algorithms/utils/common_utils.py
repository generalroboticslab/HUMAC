import socket
from contextlib import closing

from attrs import define, field


@define(order=True)
class SortedItem:
    priority: int
    item: any = field(order=False)


def find_free_port():
    """Finds an open port on the system.

    Returns:
        The open port.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
