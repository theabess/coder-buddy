"""
Sandbox backend implementations for Coder Buddy.

All backends implement the ``SandboxBackend`` abstract interface defined
in ``sandbox.base``.  The concrete backend is selected at agent
construction time via ``AgentConfig.sandbox_backend`` and injected into
the graph via dependency injection — ``Execute_Node`` never imports a
concrete backend directly.

Available backends:
- ``subprocess+venv`` — ``SubprocessVenvBackend`` (default, no extra deps)
- ``docker``          — ``DockerBackend`` (requires ``docker`` SDK)
- ``e2b``             — ``E2BBackend`` (requires ``e2b`` SDK + API key)
- ``pyodide``         — ``PyodideBackend`` (requires ``pyodide`` runtime)
"""
