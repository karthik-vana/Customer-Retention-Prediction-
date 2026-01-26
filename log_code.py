"""
LOG_CODE.PY - Logging Configuration
============================================================================
Unified logging setup for all modules in the churn prediction pipeline
============================================================================
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
log_filename = LOG_DIR / f"churn_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def setup_logging(module_name):
    """
    Setup logging for a module
    
    Args:
        module_name: Name of the module to log
        
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(module_name)
    return logger

# Log startup
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("CHURN PREDICTION PIPELINE - LOGGING INITIALIZED")
logger.info(f"Log file: {log_filename}")
logger.info("="*80)
