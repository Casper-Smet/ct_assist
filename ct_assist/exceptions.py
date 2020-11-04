from warnings import WarningMessage


class MissingExifError(Exception):
    pass


class DimensionError(Exception):
    pass


class SkipFieldWarning(UserWarning):
    pass
