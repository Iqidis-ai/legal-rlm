"""UI modules for RLM capabilities."""

# Lazy imports to avoid circular import issues
__all__ = ["create_app", "create_chat_app"]

def create_app(*args, **kwargs):
    """Create the single-turn app (lazy import)."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)

def create_chat_app(*args, **kwargs):
    """Create the multi-turn chat app with stop button (lazy import)."""
    from .chat_app import create_chat_app as _create_chat_app
    return _create_chat_app(*args, **kwargs)
