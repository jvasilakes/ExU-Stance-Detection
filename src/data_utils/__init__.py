from .util import get_datamodule  # noqa

# Populates DATASET_REGISTRY
# SuperGLUE datasets
from .datasets.axb import AXBDataset  # noqa
from .datasets.axg import AXGDataset  # noqa
from .datasets.boolq import BoolQDataset  # noqa
from .datasets.cb import CBDataset  # noqa
from .datasets.copa import COPADataset, SwappedCOPADataset  # noqa
from .datasets.rte import RTEDataset, SwappedRTEDataset  # noqa
from .datasets.wic import WiCDataset  # noqa
from .datasets.wsc import WSCDataset  # noqa

# Other datasets
from .datasets.rumoureval import RumourEvalDataset  # noqa
