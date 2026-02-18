"""
Compatibility utilities for Python versions.
"""


# Always define anext explicitly to avoid import issues
async def anext(iterator):
    """Get the next item from an async iterator."""
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        raise
