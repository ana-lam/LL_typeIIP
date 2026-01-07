import yaml
from pathlib import Path

# Project root directory (3 levels up from this file: src/lltypeiip/config/__init__.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()


class Config:
    def __init__(self, config_dict, resolve_paths=False):
        self._config = config_dict
        self._resolve_paths = resolve_paths
        self._setup_attributes()
    
    def _setup_attributes(self):
        for key, value in self._config.items():
            if isinstance(value, dict):
                # Pass resolve_paths flag to nested configs
                setattr(self, key, Config(value, resolve_paths=self._resolve_paths))
            else:
                # Resolve relative paths in the 'paths' section
                if self._resolve_paths and key.endswith('_dir') or key in ['params', 'ztf_coords', 'zenodo_meta']:
                    if isinstance(value, str) and not Path(value).is_absolute():
                        value = str(PROJECT_ROOT / value)
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __repr__(self):
        return f"Config({list(self._config.keys())})"
    
    def to_dict(self):
        return self._config.copy()
    
    def get(self, key, default=None):
        return self._config.get(key, default)


def load_config(config_path=None, resolve_paths=True):
    """
    Load configuration from YAML file.
    
    config_path : Path, optional
        Path to configuration YAML file. If None, loads defaults.yaml from this directory.
    resolve_paths : bool, optional
        If True, resolve relative paths to absolute paths using PROJECT_ROOT. Default True.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "defaults.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle path resolution specially for the 'paths' section
    if resolve_paths and 'paths' in config_dict:
        paths_dict = config_dict['paths']
        for key, value in paths_dict.items():
            if isinstance(value, str) and not Path(value).is_absolute():
                paths_dict[key] = str(PROJECT_ROOT / value)
        config_dict['paths'] = paths_dict
    
    # Handle other path-like entries (dustmaps, dusty)
    if resolve_paths:
        for section in ['dustmaps', 'dusty']:
            if section in config_dict:
                section_dict = config_dict[section]
                for key, value in section_dict.items():
                    if key.endswith('_dir') and isinstance(value, str) and not Path(value).is_absolute():
                        section_dict[key] = str(PROJECT_ROOT / value)
    
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
    'PROJECT_ROOT',
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
