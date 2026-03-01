import os
import streamlit.components.v1 as components

_RELEASE = True

if _RELEASE:
    _component_func = components.declare_component(
        "speak2list_mic",
        path=os.path.join(os.path.dirname(__file__), "frontend", "dist"),
    )
else:
    _component_func = components.declare_component(
        "speak2list_mic",
        url="http://localhost:5173",
    )

def speak2list_mic(key=None):
    return _component_func(key=key, default=None)
