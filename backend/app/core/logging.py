import logging
import logging.config
import sys
from pathlib import Path
import yaml
from backend.app.core.config import settings, PROJECT_ROOT


def setup_logging() -> None:
    """Configures application-wide logging using logging_config.yaml or fallback."""
    config_file = PROJECT_ROOT / "configs" / "logging_config.yaml"
    logs_dir = settings.resolve_path(settings.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if config_file.is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            
            # Ensure filename inside handlers points to resolved logs directory
            if "handlers" in config_dict and "file" in config_dict["handlers"]:
                config_dict["handlers"]["file"]["filename"] = str(logs_dir / "application.log")

            logging.config.dictConfig(config_dict)
            logger = logging.getLogger("app")
            logger.info("Logging configured via YAML dictionary config.")
            return
        except Exception as e:
            print(f"Warning: Failed to load logging_config.yaml ({e}), falling back to standard config.", file=sys.stderr)

    # Fallback basic configuration
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "application.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("app").info("Logging configured via fallback standard setup.")


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger."""
    return logging.getLogger(f"app.{name}")
