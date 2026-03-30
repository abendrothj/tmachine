__all__ = ["app", "celery_app"]


def __getattr__(name: str):
    """Lazy imports — avoid pulling in FastAPI/Celery until actually needed."""
    if name == "app":
        from .app import app
        return app
    if name == "celery_app":
        from .worker import celery_app
        return celery_app
    raise AttributeError(f"module 'tmachine.api' has no attribute {name!r}")
