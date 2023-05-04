from psycop.global_utils.paths import OVARTACI_SHARED_DIR

from joblib import Memory

mem = Memory(location=OVARTACI_SHARED_DIR / "cache", verbose=1)
