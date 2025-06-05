from . import coarsen  # noqa: F401
from . import constrain  # noqa: F401
from . import diff  # noqa: F401
from . import regrid  # noqa: F401
from . import remapcon  # noqa: F401
from . import resample  # noqa: F401
from . import select_domain  # noqa: F401
from . import shift_lon_break  # noqa: F401
from . import split_by_year  # noqa: F401
from . import sum  # noqa: F401
from . import vorticity  # noqa: F401
from . import actions_registry


def get_action(name):
    return actions_registry._ACTIONS[name]
