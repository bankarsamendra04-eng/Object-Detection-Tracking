from backend.app.core.config import settings, PROJECT_ROOT


def test_settings_initialization():
    assert settings.APP_NAME is not None
    assert settings.PORT > 0
    assert settings.CONFIDENCE_THRESHOLD >= 0.0
    assert settings.IOU_THRESHOLD >= 0.0


def test_effective_device():
    device = settings.get_effective_device()
    assert device in ["cuda:0", "cpu"] or device.startswith("cuda")


def test_path_resolution():
    resolved_data = settings.resolve_path(settings.DATA_DIR)
    assert resolved_data.is_absolute()
    assert str(PROJECT_ROOT) in str(resolved_data)
