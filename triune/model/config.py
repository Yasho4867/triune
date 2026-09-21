"""Canonical research configuration defaults for Triune Transformer."""

# ============================================================
# MODEL
# ============================================================

VOCAB_SIZE = 32000

HIDDEN_DIM = 1536
NUM_LAYERS = 24
NUM_HEADS = 12

# Early exits
ROUTER_PREFIX_LAYERS = 3
REFLEX_EXIT_LAYER = 6
LIMBIC_EXIT_LAYER = 16

# ============================================================
# ATTENTION
# ============================================================

USE_GLA = True
USE_ROPE = True

GLA_HEAD_DIM = 128
ROPE_MAX_SEQ_LEN = 4096

# ============================================================
# MIXTURE OF EXPERTS
# ============================================================

NUM_EXPERTS = 8
TOP_K_EXPERTS = 1

SHARED_EXPERT = True
SHARED_EXPERT_SCALE = 0.50

EXPERT_HIDDEN_MULTIPLIER = 6

MOE_CAPACITY_MULTIPLIER = 1.5

# smoother routing updates
MOE_BIAS_UPDATE_RATE = 1e-3

# centroid EMA
CENTROID_EMA = 0.98

# numerical stability
ROUTER_NOISE_STD = 0.0

# ============================================================
# DEPTH ROUTER
# ============================================================

TARGET_DEPTH_DIST = (
    0.34,
    0.33,
    0.33,
)

DEPTH_BIAS_STRENGTH = 4.0
DEPTH_BALANCE_COEF = 0.30

DEPTH_USAGE_EMA = 0.98

# ============================================================
# GALORe / Expert Optimizer
# ============================================================

GALORE = True

GALORE_RANK = 256
GALORE_UPDATE_GAP = 200

GALORE_LR = 1e-4
GALORE_BETAS = (0.9, 0.999)
GALORE_WEIGHT_DECAY = 0.05

STEER_SCALE = 0.20

# ============================================================
# MUON / Non-Expert 2D Optimizer
# ============================================================

USE_MUON = True
MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.01

# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4

SEQ_LEN = 256

TOTAL_STEPS = 50_000

LR = 1e-4
MIN_LR = 1e-6

WARMUP_STEPS = 500

BETAS = (
    0.9,
    0.95,
)

WEIGHT_DECAY = 0.05

GRAD_CLIP = 1.0

GRAD_CHECKPOINT = True

# ============================================================
# DATA
# ============================================================

SHUFFLE_BUFFER = 4096

# ============================================================
# LOGGING
# ============================================================

LOG_EVERY = 10

SAVE_EVERY = 1000

EVAL_EVERY = 200
EVAL_BATCHES = 5

CHECKPOINT_DIR = "checkpoints_full"

# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

EPS = 1e-6

ROUTER_Z_LOSS = 1e-3
