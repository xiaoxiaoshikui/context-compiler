"""Exception types shared across downstream clients."""


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass
