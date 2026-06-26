import math
import time
import logging
from typing import Tuple, Callable, Any

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)

logger = get_logger("M3_Utils")

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculates the Haversine distance between two GPS coordinates in kilometers.
    Coords must be (latitude, longitude).
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Radius of the Earth in km
    R = 6371.0

    # Convert coordinates to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c

def timer_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log the execution time of a function."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Function '{func.__name__}' completed in {elapsed:.4f} seconds.")
        return result
    return wrapper
