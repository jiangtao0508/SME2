"""Pytest probe for FlagGems backend selection and bmm registration metadata."""

import datetime as dt
import inspect
import json
import os
import sys
import threading

import pytest


_LOCK = threading.Lock()
_PATCHED = False


def _emit(event, **payload):
    path = os.environ.get("SME2_PYTEST_PROBE_LOG")
    if not path:
        return
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
    }
    row.update(payload)
    try:
        with _LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _function_info(function):
    if function is None:
        return None
    try:
        source = inspect.getsourcefile(function) or inspect.getfile(function)
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source": source,
    }


def _dispatch_table(torch_module, opname):
    try:
        return torch_module._C._dispatch_dump_table(opname)
    except (AttributeError, RuntimeError) as exc:
        return "[unavailable] {}".format(exc)


def _record_flaggems_state(stage):
    flag_gems = sys.modules.get("flag_gems")
    if flag_gems is None:
        _emit(
            "flaggems_not_loaded",
            stage=stage,
            reason="test collection did not import flag_gems",
        )
        return
    torch = sys.modules.get("torch")
    if torch is None:
        _emit(
            "torch_not_loaded",
            stage=stage,
            reason="test collection did not import torch",
        )
        return
    registrar = getattr(flag_gems, "current_work_registrar", None)
    _emit(
        "flaggems_state",
        stage=stage,
        vendor_name=getattr(flag_gems, "vendor_name", None),
        device=getattr(flag_gems, "device", None),
        version=getattr(flag_gems, "__version__", None),
        backend_info=repr(getattr(flag_gems, "backend_info", None))[:2000],
        flaggems_bmm=_function_info(getattr(flag_gems, "bmm", None)),
        torch_bmm=_function_info(getattr(torch, "bmm", None)),
        registrar_class=(
            "{}.{}".format(
                type(registrar).__module__, type(registrar).__qualname__
            )
            if registrar is not None
            else None
        ),
        aten_bmm_dispatch_table=_dispatch_table(torch, "aten::bmm"),
        aten_bmm_out_dispatch_table=_dispatch_table(torch, "aten::bmm.out"),
    )


def _patch_use_gems():
    global _PATCHED
    if _PATCHED:
        return
    flag_gems = sys.modules.get("flag_gems")
    if flag_gems is None:
        _emit("flaggems_patch_skipped", error="flag_gems is not loaded")
        return
    context_class = getattr(flag_gems, "use_gems", None)
    if context_class is None or not hasattr(context_class, "__enter__"):
        _emit("flaggems_patch_skipped", error="use_gems class not found")
        return
    original_enter = context_class.__enter__

    def traced_enter(instance):
        result = original_enter(instance)
        _record_flaggems_state("after_use_gems_enter")
        return result

    context_class.__enter__ = traced_enter
    _PATCHED = True
    _emit("flaggems_use_gems_patched")


def pytest_configure(config):
    _emit(
        "pytest_configure",
        args=[str(item) for item in config.invocation_params.args],
        rootpath=str(config.rootpath),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    _record_flaggems_state("after_collection")
    _patch_use_gems()
    _emit(
        "pytest_collection_finish",
        nodeids=[item.nodeid for item in session.items],
    )


def pytest_runtest_logstart(nodeid, location):
    _emit(
        "pytest_runtest_logstart",
        nodeid=nodeid,
        location=[str(item) for item in location],
    )


def pytest_runtest_logfinish(nodeid, location):
    _emit("pytest_runtest_logfinish", nodeid=nodeid)
