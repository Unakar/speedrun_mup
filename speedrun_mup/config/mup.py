"""
MuP-specific configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from ..models.gpt import GPTConfig


@dataclass
class ScalingConfig:
    """Configuration for model scaling experiments."""
    
    # Base model dimensions (reference for scaling)
    base_n_embd: int = 256
    base_n_layer: int = 6
    base_n_head: int = 4
    
    # Target model dimensions 
    target_n_embd: int = 768
    target_n_layer: int = 12
    target_n_head: int = 12
    
    # Delta model dimensions (for computing base shapes)
    delta_n_embd: Optional[int] = None  # If None, uses target_n_embd
    delta_n_layer: Optional[int] = None  # If None, uses target_n_layer
    delta_n_head: Optional[int] = None   # If None, uses target_n_head
    
    # Width scaling ranges for experiments
    width_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    
    # Fixed dimensions (don't scale with width)
    vocab_size: int = 50304
    block_size: int = 1024
    
    def __post_init__(self):
        # Set delta dimensions if not provided
        if self.delta_n_embd is None:
            self.delta_n_embd = self.target_n_embd
        if self.delta_n_layer is None:
            self.delta_n_layer = self.target_n_layer
        if self.delta_n_head is None:
            self.delta_n_head = self.target_n_head
            
        # Validate that dimensions are consistent
        assert self.base_n_embd % self.base_n_head == 0
        assert self.target_n_embd % self.target_n_head == 0
        assert self.delta_n_embd % self.delta_n_head == 0
    
    def get_width_mult(self) -> float:
        """Get the width multiplier from base to target."""
        return self.target_n_embd / self.base_n_embd
    
    def get_base_config(self) -> GPTConfig:
        """Get GPT configuration for base model."""
        return GPTConfig(
            vocab_size=self.vocab_size,
            n_layer=self.base_n_layer,
            n_head=self.base_n_head,
            n_embd=self.base_n_embd,
            block_size=self.block_size,
        )
    
    def get_target_config(self) -> GPTConfig:
        """Get GPT configuration for target model."""
        return GPTConfig(
            vocab_size=self.vocab_size,
            n_layer=self.target_n_layer,
            n_head=self.target_n_head,
            n_embd=self.target_n_embd,
            block_size=self.block_size,
        )
    
    def get_delta_config(self) -> GPTConfig:
        """Get GPT configuration for delta model."""
        return GPTConfig(
            vocab_size=self.vocab_size,
            n_layer=self.delta_n_layer,
            n_head=self.delta_n_head,
            n_embd=self.delta_n_embd,
            block_size=self.block_size,
        )
    
    def get_scaled_config(self, width_mult: float) -> GPTConfig:
        """Get GPT configuration for a specific width multiplier."""
        scaled_n_embd = int(self.base_n_embd * width_mult)
        scaled_n_head = max(1, int(self.base_n_head * width_mult**0.5))  # Scale heads more slowly
        
        # Ensure n_embd is divisible by n_head
        while scaled_n_embd % scaled_n_head != 0:
            scaled_n_head -= 1
        
        return GPTConfig(
            vocab_size=self.vocab_size,
            n_layer=self.base_n_layer,  # Keep depth fixed
            n_head=scaled_n_head,
            n_embd=scaled_n_embd,
            block_size=self.block_size,
        )


@dataclass
class MuPConfig:
    """Configuration for MuP parameterization settings."""
    
    # MuP activation
    use_mup: bool = True
    
    # Coordinate checking
    coord_check_enabled: bool = True
    coord_check_nsteps: int = 3
    coord_check_nseeds: int = 1
    
    # Base shapes
    base_shapes_file: Optional[str] = None
    save_base_shapes: bool = True
    
    # Learning rate scaling
    base_lr: float = 3e-4
    lr_scale_output: bool = True  # Scale output layer LR differently
    lr_scale_embeddings: bool = True  # Scale embedding LR differently
    
    # Weight decay scaling  
    base_weight_decay: float = 0.1
    scale_weight_decay: bool = True
    
    # Initialization scaling
    scale_init: bool = True
    init_std: float = 0.02
    
    # Advanced MuP features
    spectral_monitoring: bool = False
    higher_order_mup: bool = False
    
    # Validation thresholds
    coord_check_tolerance: float = 2.0  # Max allowed coordinate scaling
    activation_tolerance: float = 5.0   # Max allowed activation magnitude
    
    def get_lr_multipliers(self, width_mult: float) -> Dict[str, float]:
        """Get learning rate multipliers for different parameter types."""
        multipliers = {
            'default': 1.0,  # Base learning rate
            'matrix': 1.0 / width_mult,  # Matrix parameters: LR scales as 1/width
            'vector': width_mult,        # Vector parameters: LR scales as width
            'embedding': 1.0,           # Embeddings: keep base LR
            'output': 1.0 / width_mult, # Output layer: scale as 1/width
        }
        
        if not self.lr_scale_output:
            multipliers['output'] = 1.0
        if not self.lr_scale_embeddings:
            multipliers['embedding'] = 1.0
            
        return multipliers
    
    def get_wd_multipliers(self, width_mult: float) -> Dict[str, float]:
        """Get weight decay multipliers for different parameter types."""
        if not self.scale_weight_decay:
            return {'default': 1.0}
        
        return {
            'default': 1.0,
            'matrix': width_mult,     # Matrix parameters: WD scales as width
            'vector': 1.0,           # Vector parameters: keep base WD
            'embedding': 1.0,        # Embeddings: keep base WD
            'output': width_mult,    # Output layer: scale as width
        }


@dataclass
class ValidationConfig:
    """Configuration for MuP validation and testing."""
    
    # Coordinate checking
    coord_check_widths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    coord_check_layers: List[str] = field(default_factory=lambda: ['transformer.h.0.attn', 'transformer.h.6.mlp'])
    
    # Scaling tests
    scaling_test_widths: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    scaling_test_steps: int = 1000
    
    # Hyperparameter transfer tests
    transfer_base_width: int = 256
    transfer_target_widths: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    transfer_steps: int = 5000
    
    # Spectral analysis
    spectral_layers: List[str] = field(default_factory=lambda: ['transformer.h.0', 'transformer.h.6', 'transformer.h.-1'])
    spectral_frequency: int = 500  # Steps between spectral analysis
    spectral_top_k: int = 5        # Number of top singular values to track
    
    # Tolerances for validation
    coord_tolerance: float = 2.0
    activation_tolerance: float = 5.0
    spectral_tolerance: Tuple[float, float] = (0.1, 10.0)  # (min, max) acceptable spectral norm
    
    def should_run_coord_check(self, step: int) -> bool:
        """Check if coordinate checking should run at this step."""
        return step <= 10  # Run for first few steps only
    
    def should_run_spectral(self, step: int) -> bool:
        """Check if spectral analysis should run at this step."""
        return step % self.spectral_frequency == 0