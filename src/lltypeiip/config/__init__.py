import yaml
from pathlib import Path


class Config:
    def __init__(self, config_dict):
        self._config = config_dict
        self._setup_attributes()
    
    def _setup_attributes(self):
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __repr__(self):
        return f"Config({list(self._config.keys())})"
    
    def to_dict(self):
        return self._config.copy()
    
    def get(self, key, default=None):
        return self._config.get(key, default)


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    config_path : Path, optional
        Path to configuration YAML file. If None, loads defaults.yaml from this directory.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "defaults.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def setup_dustmaps(config_obj=None):
    import dustmaps.config
    from dustmaps.sfd import SFDQuery
    
    if config_obj is None:
        config_obj = config
    
    data_dir = config_obj.dustmaps.data_dir
    sfd_dir = config_obj.dustmaps.sfd_dir
    
    dustmaps.config.config['data_dir'] = data_dir
    
    sfd = SFDQuery(map_dir=sfd_dir)

    return sfd

# Load default configuration on import
config = load_config()

SNR_MIN = config.snr.min
SNR_MIN_WISE = config.snr.min_wise

LAM_EFF = config.wavelengths.to_dict()

SED_COLORS = config.sed_plot.colors.to_dict()
SED_MARKERS = config.sed_plot.markers.to_dict()
SED_SIZE = config.sed_plot.size

EXTINCTION_RV = config.extinction.r_v
EXTINCTION_SF11_SCALE = config.extinction.sf11_scale
EXTINCTION_COEFFS = config.extinction.cardelli_coefficients.to_dict()

__all__ = [
    'config',
    'load_config',
    'setup_dustmaps',
    'Config',
    'SNR_MIN',
    'SNR_MIN_WISE',
    'LAM_EFF',
    'SED_COLORS',
    'SED_MARKERS',
    'SED_SIZE',
    'EXTINCTION_RV',
    'EXTINCTION_SF11_SCALE',
    'EXTINCTION_COEFFS',
]
