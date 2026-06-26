import os

# Base paths
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(CONFIG_DIR, "output")

# Centrality Analysis Settings
DEFAULT_K_CENTRALITY = 50

# Simulation Settings
DEFAULT_DISASTER_FRACTION = 0.05
DEFAULT_ROADBLOCK_FRACTION = 0.01

# AI Decision Support (Groq) Configuration
GROQ_MODEL_NAME = "llama3-8b-8192"

# Export Metadata
METADATA_MODULE = "Member3"
METADATA_VERSION = "1.0"
