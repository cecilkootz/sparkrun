"""sparkrun proxy commands — unified OpenAI-compatible gateway."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_node_trimming,
    _apply_recipe_overrides,
    _load_recipe,
    _resolve_cluster_cache_dir,
    _resolve_hosts_or_exit,
    _resolve_transfer_mode,
    dry_run_option,
    host_options,
    recipe_override_options,
)


@click.group()
def proxy():
    """Manage the LiteLLM-based inference proxy.

    The proxy discovers running sparkrun inference endpoints and
    presents them through a single unified OpenAI-compatible API.
    """


# ---------------------------------------------------------------------------
# proxy start
# ---------------------------------------------------------------------------

@proxy.command()
@click.option("--port", type=int, default=None, help="Proxy listen port (default: 4000)")
@click.option("--host", "bind_host", default=None, help="Bind address (default: 0.0.0.0)")
@click.option("--master-key", default=None, help="LiteLLM master_key (default: sk-sparkrun)")
@host_options
@click.option("--foreground", is_flag=True, help="Run in foreground (default: daemonize)")
@click.option("--no-auto-discover", is_flag=True, help="Disable periodic endpoint re-scanning")
@click.option("--discover-interval", type=int, default=None,
              help="Seconds between discovery sweeps (default: 30)")
@dry_run_option
def start(port, bind_host, master_key, cluster_name, hosts, hosts_file,
          foreground, no_auto_discover, discover_interval, dry_run):
    """Start the inference proxy.

    Discovers running endpoints, generates LiteLLM config, and launches
    the proxy via ``uvx litellm``.

    Examples:

      sparkrun proxy start

      sparkrun proxy start --cluster mylab --port 4000

      sparkrun proxy start --foreground
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.discovery import discover_endpoints
    from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

    proxy_cfg = ProxyConfig()

    effective_port = port or proxy_cfg.port
    effective_host = bind_host or proxy_cfg.host
    effective_key = master_key if master_key is not None else proxy_cfg.master_key

    # Resolve host filter
    host_filter = _resolve_host_filter(cluster_name, hosts, hosts_file)

    # Discover endpoints
    click.echo("Discovering inference endpoints...")
    endpoints = discover_endpoints(host_filter=host_filter)

    healthy = [ep for ep in endpoints if ep.healthy]
    if not healthy:
        if endpoints:
            click.echo("Found %d endpoint(s) but none are healthy." % len(endpoints))
        else:
            click.echo("No inference endpoints found.")
        click.echo("Start inference workloads first: sparkrun run <recipe>")
        if not dry_run:
            sys.exit(1)
        return

    click.echo("Discovered %d healthy endpoint(s):" % len(healthy))
    for ep in healthy:
        models_str = ", ".join(ep.actual_models) if ep.actual_models else ep.model
        click.echo("  %s:%d — %s (%s)" % (ep.host, ep.port, models_str, ep.runtime))

    # Generate config
    aliases = proxy_cfg.aliases
    config_dict = build_litellm_config(healthy, aliases, effective_key)

    if dry_run:
        click.echo("")
        click.echo("[dry-run] Would write litellm config and start proxy on %s:%d"
                   % (effective_host, effective_port))
        if aliases:
            click.echo("[dry-run] Aliases: %s" % aliases)
        return

    config_path = write_config(config_dict)
    click.echo("")

    # Launch proxy
    engine = ProxyEngine(
        host=effective_host,
        port=effective_port,
        master_key=effective_key,
    )

    click.echo("Starting proxy on %s:%d..." % (effective_host, effective_port))
    rc = engine.start(config_path=config_path, foreground=foreground)
    if rc != 0:
        click.echo("Error: proxy failed to start (exit code %d)" % rc, err=True)
        sys.exit(rc)

    if not foreground:
        click.echo("Proxy started. API: http://localhost:%d/v1" % effective_port)
        if effective_key:
            click.echo("Management API key: %s" % effective_key)
        # click.echo("Log: %s" % (engine.state_dir / "litellm.log"))


# ---------------------------------------------------------------------------
# proxy stop
# ---------------------------------------------------------------------------

@proxy.command()
@dry_run_option
def stop(dry_run):
    """Stop the running proxy.

    Sends SIGTERM to the proxy process using the stored PID.
    """
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()

    if not engine.is_running():
        click.echo("No proxy is currently running.")
        return

    if engine.stop(dry_run=dry_run):
        click.echo("Proxy stopped.")
    else:
        click.echo("Failed to stop proxy.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# proxy status
# ---------------------------------------------------------------------------

@proxy.command()
def status():
    """Show proxy process status and registered models."""
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()
    state = engine.get_state()

    if not state:
        click.echo("No proxy state found.")
        return

    running = engine.is_running()
    click.echo("Proxy status: %s" % ("running" if running else "stopped (stale state)"))
    click.echo("  PID:    %s" % state.get("pid", "?"))
    click.echo("  Host:   %s" % state.get("host", "?"))
    click.echo("  Port:   %s" % state.get("port", "?"))
    click.echo("  Start:  %s" % state.get("started_at", "?"))

    if running:
        models = engine.list_models_via_api()
        if models:
            click.echo("")
            click.echo("Registered models (%d):" % len(models))
            for m in models:
                name = m.get("model_name", "?")
                info = m.get("model_info", {})
                click.echo("  %s" % name)
                if info.get("litellm_params", {}).get("api_base"):
                    click.echo("    -> %s" % info["litellm_params"]["api_base"])
        else:
            click.echo("")
            click.echo("No models registered (or management API unavailable).")


# ---------------------------------------------------------------------------
# proxy discover
# ---------------------------------------------------------------------------

@proxy.command()
@host_options
@click.option("--no-health-check", is_flag=True, help="Skip health checks")
def discover(hosts, hosts_file, cluster_name, no_health_check):
    """One-shot endpoint discovery (debug/inspection).

    Scans job metadata and optionally health-checks each endpoint.
    Does not start the proxy.

    Examples:

      sparkrun proxy discover

      sparkrun proxy discover --cluster mylab

      sparkrun proxy discover --no-health-check
    """
    from sparkrun.proxy.discovery import discover_endpoints

    host_filter = _resolve_host_filter(cluster_name, hosts, hosts_file)

    endpoints = discover_endpoints(
        host_filter=host_filter,
        check_health=not no_health_check,
    )

    if not endpoints:
        click.echo("No inference endpoints found in job metadata.")
        return

    click.echo("Discovered %d endpoint(s):" % len(endpoints))
    click.echo("")
    for ep in endpoints:
        health = "healthy" if ep.healthy else "unreachable"
        if no_health_check:
            health = "unchecked"
        models_str = ", ".join(ep.actual_models) if ep.actual_models else ep.model
        click.echo("  %-20s %s:%d" % (ep.cluster_id, ep.host, ep.port))
        click.echo("    Recipe:   %s" % ep.recipe_name)
        click.echo("    Model:    %s" % models_str)
        click.echo("    Runtime:  %s" % ep.runtime)
        click.echo("    TP:       %d" % ep.tensor_parallel)
        click.echo("    Status:   %s" % health)
        if ep.served_model_name:
            click.echo("    Served:   %s" % ep.served_model_name)
        click.echo("")


# ---------------------------------------------------------------------------
# proxy models
# ---------------------------------------------------------------------------

@proxy.command()
@click.option("--refresh", is_flag=True, help="Re-discover endpoints and update proxy")
def models(refresh):
    """List models registered with the proxy.

    Uses the management API to query the running proxy.
    With --refresh, re-discovers endpoints and adds new models.
    """
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()

    if not engine.is_running():
        click.echo("Proxy is not running. Start it with: sparkrun proxy start")
        return

    if refresh:
        from sparkrun.proxy.discovery import discover_endpoints
        click.echo("Re-discovering endpoints...")
        endpoints = discover_endpoints()
        healthy = [ep for ep in endpoints if ep.healthy]
        added, removed = engine.sync_models(healthy)
        if added or removed:
            parts = []
            if added:
                parts.append("added %d" % added)
            if removed:
                parts.append("removed %d stale" % removed)
            click.echo("Synced proxy models: %s." % ", ".join(parts))
        else:
            click.echo("Proxy models already in sync.")

    model_list = engine.list_models_via_api()
    if not model_list:
        click.echo("No models registered with the proxy.")
        return

    click.echo("Models (%d):" % len(model_list))
    for m in model_list:
        name = m.get("model_name", "?")
        params = m.get("litellm_params", m.get("model_info", {}).get("litellm_params", {}))
        api_base = params.get("api_base", "?")
        click.echo("  %-40s -> %s" % (name, api_base))


# ---------------------------------------------------------------------------
# proxy alias
# ---------------------------------------------------------------------------

@proxy.group()
def alias():
    """Manage model aliases."""


@alias.command("add")
@click.argument("alias_name")
@click.argument("target_model")
def alias_add(alias_name, target_model):
    """Add a model alias.

    ALIAS_NAME is the friendly name clients will use.
    TARGET_MODEL is the actual model group name to route to.

    Example:

      sparkrun proxy alias add my-model "Qwen/Qwen3-1.7B"
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.engine import ProxyEngine

    proxy_cfg = ProxyConfig()
    proxy_cfg.add_alias(alias_name, target_model)
    proxy_cfg.save()
    click.echo("Alias added: %s -> %s" % (alias_name, target_model))

    # Reload proxy if running
    engine = ProxyEngine()
    if engine.is_running():
        click.echo("Reloading proxy to apply alias...")
        from sparkrun.proxy.discovery import discover_endpoints
        endpoints = discover_endpoints()
        healthy = [ep for ep in endpoints if ep.healthy]
        rc = engine.reload(healthy, proxy_cfg.aliases)
        if rc == 0:
            click.echo("Proxy reloaded.")
        else:
            click.echo("Warning: proxy reload failed (exit code %d)" % rc, err=True)


@alias.command("remove")
@click.argument("alias_name")
def alias_remove(alias_name):
    """Remove a model alias.

    Example:

      sparkrun proxy alias remove my-model
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.engine import ProxyEngine

    proxy_cfg = ProxyConfig()
    if not proxy_cfg.remove_alias(alias_name):
        click.echo("Alias '%s' not found." % alias_name)
        return

    proxy_cfg.save()
    click.echo("Alias removed: %s" % alias_name)

    # Reload proxy if running
    engine = ProxyEngine()
    if engine.is_running():
        click.echo("Reloading proxy to apply changes...")
        from sparkrun.proxy.discovery import discover_endpoints
        endpoints = discover_endpoints()
        healthy = [ep for ep in endpoints if ep.healthy]
        rc = engine.reload(healthy, proxy_cfg.aliases)
        if rc == 0:
            click.echo("Proxy reloaded.")
        else:
            click.echo("Warning: proxy reload failed (exit code %d)" % rc, err=True)


@alias.command("list")
def alias_list():
    """List all configured aliases."""
    from sparkrun.proxy.config import ProxyConfig

    proxy_cfg = ProxyConfig()
    aliases = proxy_cfg.list_aliases()

    if not aliases:
        click.echo("No aliases configured.")
        click.echo("Add one with: sparkrun proxy alias add <name> <model>")
        return

    click.echo("Aliases:")
    for name, target in aliases:
        click.echo("  %-30s -> %s" % (name, target))


# ---------------------------------------------------------------------------
# proxy load / unload
# ---------------------------------------------------------------------------

@proxy.command("load")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory")
@dry_run_option
def load_cmd(recipe_name, hosts, hosts_file, cluster_name,
             tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
             options, image, solo, port, cache_dir, dry_run):
    """Load a model via sparkrun run and register with proxy.

    Launches inference and registers the new endpoint with the running
    proxy via the management API.

    Example:

      sparkrun proxy load qwen3-1.7b-vllm --cluster mylab

      sparkrun proxy load qwen3-1.7b-vllm --solo --gpu-mem 0.8
    """
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference

    v = init_sparkrun()
    config = SparkrunConfig()

    # Load recipe
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    issues = recipe.validate()
    for issue in issues:
        click.echo("Warning: %s" % issue, err=True)

    # Build overrides
    overrides = _apply_recipe_overrides(
        options, tensor_parallel=tensor_parallel, pipeline_parallel=pipeline_parallel,
        gpu_mem=gpu_mem, max_model_len=max_model_len, image=image, recipe=recipe,
    )
    if port is not None:
        overrides["port"] = port

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Resolve hosts
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

    # Node count validation / trimming
    if len(host_list) > 1 and not solo:
        try:
            required = runtime.compute_required_nodes(recipe, overrides)
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        if required is not None:
            if required > len(host_list):
                click.echo(
                    "Error: runtime requires %d nodes, but only %d hosts provided"
                    % (required, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif required < len(host_list):
                host_list = _apply_node_trimming(
                    host_list, recipe, overrides, runtime=runtime,
                )

    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        host_list = host_list[:recipe.max_nodes]

    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "solo":
        is_solo = True
    if is_solo and len(host_list) > 1:
        host_list = host_list[:1]

    # Resolve cache dir and transfer mode
    cluster_cache_dir = _resolve_cluster_cache_dir(cluster_name, hosts, hosts_file, cluster_mgr)
    effective_cache_dir = cache_dir or cluster_cache_dir or str(config.hf_cache_dir)
    cluster_transfer_mode = _resolve_transfer_mode(cluster_name, hosts, hosts_file, cluster_mgr)
    effective_transfer_mode = cluster_transfer_mode or "auto"

    # Launch via shared pipeline (auto_port=True for conflict avoidance)
    click.echo("Loading model: %s" % recipe_name)
    result = launch_inference(
        recipe=recipe,
        runtime=runtime,
        host_list=host_list,
        overrides=overrides,
        config=config,
        v=v,
        is_solo=is_solo,
        cache_dir=effective_cache_dir,
        transfer_mode=effective_transfer_mode,
        registry_mgr=registry_mgr,
        auto_port=True,
        dry_run=dry_run,
        detached=True,
    )

    if result.rc != 0:
        click.echo("Error: failed to load model (exit code %d)." % result.rc, err=True)
        sys.exit(1)

    click.echo("Model loaded: %s (port %d)" % (recipe_name, result.serve_port))

    if not dry_run:
        # Try to register with running proxy
        from sparkrun.proxy.engine import ProxyEngine
        engine = ProxyEngine()
        if engine.is_running():
            click.echo("Registering with proxy...")
            from sparkrun.proxy.discovery import discover_endpoints
            import time
            time.sleep(2)  # Brief delay for server startup
            endpoints = discover_endpoints()
            healthy = [ep for ep in endpoints if ep.healthy]
            added = 0
            for ep in healthy:
                if engine.add_model_via_api(ep):
                    added += 1
            if added:
                click.echo("Registered %d model(s) with proxy." % added)


@proxy.command("unload")
@click.argument("recipe_name")
@host_options
@dry_run_option
def unload_cmd(recipe_name, hosts, hosts_file, cluster_name, dry_run):
    """Unload a model via sparkrun stop and remove from proxy.

    Example:

      sparkrun proxy unload qwen3-1.7b-vllm --cluster mylab
    """
    from sparkrun.proxy.loader import unload_model

    ok = unload_model(
        recipe_name=recipe_name,
        cluster=cluster_name,
        hosts=hosts,
        hosts_file=hosts_file,
        dry_run=dry_run,
    )

    if not ok:
        click.echo("Error: failed to unload model.", err=True)
        sys.exit(1)

    click.echo("Model unloaded: %s" % recipe_name)

    if not dry_run:
        # Sync proxy to remove the now-stale model entry
        from sparkrun.proxy.engine import ProxyEngine
        engine = ProxyEngine()
        if engine.is_running():
            click.echo("Syncing proxy models...")
            from sparkrun.proxy.discovery import discover_endpoints
            endpoints = discover_endpoints()
            healthy = [ep for ep in endpoints if ep.healthy]
            _added, removed = engine.sync_models(healthy)
            if removed:
                click.echo("Removed %d stale model(s) from proxy." % removed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_host_filter(
        cluster_name: str | None,
        hosts: str | None,
        hosts_file: str | None,
) -> list[str] | None:
    """Resolve host filter from CLI args without exiting on empty.

    Unlike ``_resolve_hosts_or_exit``, returns None (no filter) when
    no host source is specified — discovery will scan all job metadata.
    """
    if hosts:
        return [h.strip() for h in hosts.split(",") if h.strip()]

    if hosts_file:
        try:
            from pathlib import Path
            text = Path(hosts_file).read_text()
            return [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except OSError:
            click.echo("Warning: could not read hosts file: %s" % hosts_file, err=True)
            return None

    if cluster_name:
        try:
            from sparkrun.cli._common import _get_cluster_manager
            cluster_mgr = _get_cluster_manager()
            cluster_def = cluster_mgr.get(cluster_name)
            return cluster_def.hosts if cluster_def else None
        except Exception:
            click.echo("Warning: could not resolve cluster '%s'" % cluster_name, err=True)
            return None

    return None



