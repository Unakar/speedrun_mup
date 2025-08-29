"""
Base configuration classes for experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Optimization
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Schedule
    warmup_iters: int = 2000
    max_iters: int = 100000
    lr_decay_iters: Optional[int] = None  # If None, uses max_iters
    min_lr: float = 0.0
    
    # Training dynamics
    batch_size: int = 8
    micro_batch_size: int = 1  # For gradient accumulation
    block_size: int = 1024
    
    # Regularization
    dropout: float = 0.0
    
    # Mixed precision
    dtype: str = 'bfloat16'  # 'float16', 'bfloat16', 'float32'
    compile_model: bool = True
    
    # Checkpointing
    save_interval: int = 5000
    eval_interval: int = 1000
    log_interval: int = 100
    
    # Resuming
    resume_from: Optional[str] = None
    init_from: str = 'scratch'  # 'scratch', 'resume', 'gpt2*'
    
    def __post_init__(self):
        if self.lr_decay_iters is None:
            self.lr_decay_iters = self.max_iters
        
        # Validate gradient accumulation
        assert self.batch_size % self.micro_batch_size == 0
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size


@dataclass  
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Dataset
    dataset: str = 'fineweb'
    dataset_path: Optional[str] = None
    
    # Tokenization
    tokenizer: str = 'gpt2'
    vocab_size: int = 50304  # Padded to multiple of 128
    
    # Processing
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Data splits
    train_split: float = 0.99
    val_split: float = 0.01
    
    # Caching
    use_cached: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        assert abs(self.train_split + self.val_split - 1.0) < 1e-6


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = 'speedrun-mup'
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # File logging
    log_dir: str = './logs'
    save_logs: bool = True
    
    # Metrics
    log_train_loss: bool = True
    log_val_loss: bool = True
    log_grad_norm: bool = True
    log_weight_norm: bool = True
    log_lr: bool = True
    log_tokens_per_sec: bool = True
    
    # MuP-specific metrics
    log_activations: bool = True
    log_coordinates: bool = True
    log_spectral: bool = False
    
    # Frequency
    log_interval: int = 100
    activation_log_interval: int = 1000
    spectral_log_interval: int = 5000


@dataclass
class ExperimentConfig:
    """Main configuration that combines all components."""
    
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig) 
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Experiment metadata
    name: str = 'mup_experiment'
    description: str = ''
    tags: List[str] = field(default_factory=list)
    
    # Output
    out_dir: str = './outputs'
    
    # Reproducibility
    seed: int = 1337
    deterministic: bool = True
    
    # Hardware
    device: str = 'cuda'
    backend: str = 'nccl'
    
    def __post_init__(self):
        # Ensure output directory exists
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Extract sub-configurations
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Remove sub-configs from main dict
        main_dict = {k: v for k, v in config_dict.items() 
                    if k not in ['training', 'data', 'logging']}
        
        return cls(
            training=training_config,
            data=data_config,
            logging=logging_config,
            **main_dict
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def get_run_name(self) -> str:
        """Generate a run name for logging."""
        if self.logging.wandb_run_name:
            return self.logging.wandb_run_name
        
        # Generate based on key parameters
        lr = self.training.learning_rate
        batch_size = self.training.batch_size
        return f"{self.name}_lr{lr}_bs{batch_size}"