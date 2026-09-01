#!/usr/bin/env python3
"""
Do the pins in requirements.txt actually INSTALL on the Docker target interpreter?

`docker build` runs `pip install -r backend/requirements.txt` inside python:3.13-slim
(Debian, so manylinux wheels). The failure mode this catches is not "wrong version", it
is "no wheel": a pin that only has an sdist for cp313 turns the build into a source
compile inside a slim image with no toolchain, and a pin whose wheel is built for another
interpreter/platform simply errors out. Until now the only way to find that out was a
build that takes ten minutes to fail.

    python3 backend/check_requirements.py                       # target = cp313 / manylinux x86_64
    python3 backend/check_requirements.py --python-version 312  # check another interpreter
    python3 backend/check_requirements.py --offline             # parse only, no network

Exit status: 0 = every requirement resolvable to a usable wheel, 1 = at least one
MISSING / NO_WHEEL, 2 = the requirements file itself could not be parsed.
SDIST-only pins are reported as BUILD and do NOT fail the run: they usually compile fine,
they are just slow and fragile — treat them as a warning.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from typing import Iterable

UA = {"User-Agent": "voice-chat-check-requirements"}

# Wheels that are pure python / pure-abi and therefore always usable.
_ANY_ABIS = ("none", "abi3")


def parse_requirements(text: str) -> tuple[list[dict], list[str]]:
    """Return (requirements, find_links). Comments, blank lines and option lines are
    handled the way pip handles them; environment markers are kept so a darwin-only
    pin can be reported but not counted against the linux target."""
    reqs: list[dict] = []
    links: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("--find-links") or line.startswith("-f "):
            links.append(line.split(None, 1)[1].strip() if "=" not in line
                         else line.split("=", 1)[1].strip())
            continue
        if line.startswith("-"):                                  # --index-url, -i, …
            continue
        marker = None
        if ";" in line:
            line, marker = line.split(";", 1)
            line, marker = line.strip(), marker.strip()
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:\[([^\]]*)\])?\s*(.*)$", line)
        if not m:
            raise ValueError(f"unparseable requirement line: {raw!r}")
        name, extras, spec = m.group(1), m.group(2) or "", m.group(3).strip()
        reqs.append({"name": name, "extras": extras, "spec": spec, "marker": marker,
                     "raw": raw.strip()})
    return reqs, links


def version_key(v: str) -> tuple:
    """Loose version ordering good enough to pick a newest candidate (no deps)."""
    parts = re.findall(r"\d+|[A-Za-z]+", v)
    out: list = []
    for p in parts:
        out.append((0, int(p)) if p.isdigit() else (1, p))
    return tuple(out)


def pick_version(spec: str, available: Iterable[str]) -> str | None:
    """Pick the version pip would most plausibly install for this pin, or None."""
    spec = spec.replace(" ", "")
    # An exact pin is trusted and verified later by fetching that version's file list —
    # it may be a local version (`0.3.1+cu124`) that only exists in a --find-links index,
    # so "not in this release list" must not short-circuit it to MISSING.
    if spec.startswith("==") and not spec.endswith(".*"):
        return spec[2:] or None
    avail = [v for v in available if "+" not in v and "rc" not in v.lower() and not re.search(r"\d+b\d", v)]
    if not avail:
        return None
    if not spec:
        return max(avail, key=version_key)
    if spec.startswith("==") and spec.endswith(".*"):
        prefix = spec[2:-2]
        cand = [v for v in avail if v == prefix or v.startswith(prefix + ".")]
        return max(cand, key=version_key) if cand else None
    if spec.startswith(">="):
        want = spec[2:]
        cand = [v for v in avail if version_key(v) >= version_key(want)] or avail
        return max(cand, key=version_key)
    return max(avail, key=version_key)          # ~=, >, != … : newest is a fine proxy


# Platform tags that name a CPU architecture. Without this check an aarch64-only wheel
# looks like a match for an x86_64 build — which is exactly the false green light this
# script exists to prevent (the TTS CUDA wheel index ships both).
ARCHES = ("x86_64", "aarch64", "armv7l", "armv6l", "i686", "ppc64le", "s390x", "riscv64")


def wheel_matches(fn: str, pyver: str = "313", platforms: Iterable[str] = ("manylinux", "musllinux"),
                  arch: str | None = "x86_64") -> bool:
    """Does this wheel filename install on cp{pyver} on one of `platforms` (and `arch`)?"""
    low = fn.lower()
    if not low.endswith(".whl"):
        return False
    tags = low[:-4].split("-")
    if len(tags) < 3:
        return False
    py, abi, plat = tags[-3], tags[-2], tags[-1]
    if abi == "abi3":
        # Stable ABI: a cpNN-abi3 wheel is installable on any cp >= NN (psutil ships
        # cp36-abi3-manylinux wheels, which is exactly how cp313 gets it).
        if not (py.startswith("cp") and py[2:].isdigit() and int(py[2:]) <= int(pyver)):
            return False
    elif py not in (f"cp{pyver}", "py3", "py2.py3"):
        return False                                       # other interpreter
    elif abi != f"cp{pyver}" and abi != "none":
        return False                                       # non-stable ABI mismatch
    if plat == "any":
        return True
    if not any(plat_token in plat for plat_token in platforms):
        return False
    if arch:
        named = [a for a in ARCHES if a in plat]
        if named and arch not in named:
            return False                    # e.g. manylinux_2_35_aarch64 on an x86_64 target
    return True


def classify(files: list[dict], pyver: str, platforms: Iterable[str], arch: str | None = "x86_64") -> tuple[str, str, str]:
    """-> (status, detail, version_files) where status is OK / NO_WHEEL / SDIST."""
    platforms = list(platforms)
    for f in files:
        if wheel_matches(f.get("filename", ""), pyver, platforms, arch):
            return "OK", f["filename"], ""
    wheels = [f["filename"] for f in files if f.get("filename", "").endswith(".whl")]
    if wheels:
        return "NO_WHEEL", ", ".join(sorted({re.sub(r'^.*?-([^-]+-[^-]+-[^.]+)\.whl$', r'\1', w) for w in wheels}))[:120], ""
    return "SDIST", ", ".join(f["filename"] for f in files)[:120], ""


def pypi_json(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def hf_find_links_wheels(link: str, timeout: float = 20.0) -> list[str]:
    """`--find-links` pointing at an HF dataset folder: list its wheel filenames via the
    hub API (the HTML page is a JS app, the API is stable)."""
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/]+)/tree/main/(.*)$", link)
    if not m:
        return []
    repo, path = m.group(1), m.group(2).strip("/")
    tree = pypi_json(f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}", timeout)
    return [e["path"].rsplit("/", 1)[-1] for e in tree if e.get("path", "").endswith(".whl")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--requirements", default=None)
    ap.add_argument("--python-version", default="313", help="target interpreter, e.g. 313 (default)")
    ap.add_argument("--platform", default="manylinux,musllinux",
                    help="comma list of accepted wheel platform tokens")
    ap.add_argument("--arch", default="x86_64",
                    help="target CPU architecture (aarch64 for Graveline/Arm hosts, '' to ignore)")
    ap.add_argument("--offline", action="store_true", help="parse only, no network")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    default = __file__.rsplit("/", 1)[0] + "/requirements.txt"
    path = args.requirements or default
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"cannot read {path}: {e}")
        return 2
    try:
        reqs, links = parse_requirements(text)
    except ValueError as e:
        print(str(e))
        return 2

    platforms = tuple(p.strip() for p in args.platform.split(",") if p.strip())
    arch = args.arch.strip() or None
    print(f"{path}: {len(reqs)} requirements, target cp{args.python_version} / {'|'.join(platforms)} / {arch or 'any arch'}")
    if links and not args.offline:
        for lk in links:
            try:
                names = hf_find_links_wheels(lk)
                print(f"find-links {lk}: {len(names)} wheels"
                      f"{' e.g. ' + names[0] if names else ''}")
            except Exception as e:
                print(f"find-links {lk}: UNREACHABLE ({e!r})")
    if args.offline:
        for r in reqs:
            print(f"  PARSE  {r['name']:24} {r['spec'] or '(unpinned)':12} {('marker: ' + r['marker']) if r['marker'] else ''}")
        return 0

    link_wheels: list[str] = []
    for lk in links:
        try:
            link_wheels += hf_find_links_wheels(lk)
        except Exception:
            pass

    bad = 0
    warn = 0
    for r in reqs:
        name, spec = r["name"], r["spec"]
        if r["marker"] and "darwin" in r["marker"]:
            print(f"  SKIP   {name:24} ({r['marker']})")
            continue
        try:
            meta = pypi_json(f"https://pypi.org/pypi/{name}/json")
        except Exception as e:
            # Not on PyPI: it may come from a --find-links index instead.
            from_links = [w for w in link_wheels if w.lower().startswith(name.lower().replace("-", "_") + "-")]
            if from_links:
                ok = [w for w in from_links if wheel_matches(w, args.python_version, platforms, arch)]
                status = "OK" if ok else "NO_WHEEL"
                print(f"  {status:6} {name:24} {spec:12} from find-links: {(ok or from_links)[0]}")
                bad += 0 if ok else 1
            else:
                print(f"  MISSING {name:24} {spec:12} not on PyPI ({e.__class__.__name__}) and not in find-links")
                bad += 1
            continue
        ver = pick_version(spec, list(meta.get("releases", {}).keys()))
        if ver is None:
            print(f"  MISSING {name:24} {spec:12} no released version matches the pin")
            bad += 1
            continue
        try:
            files = pypi_json(f"https://pypi.org/pypi/{name}/{ver}/json").get("urls", [])
        except Exception:
            # Local-version pins (`pkg==X+cu124`) live only in the --find-links index,
            # which is how the TTS CUDA wheel ships; PyPI 404s by construction.
            want = (name.lower().replace("-", "_") + "-" + ver).lower()
            from_links = [w for w in link_wheels if w.lower().startswith(want)]
            if not from_links:
                from_links = [w for w in link_wheels if w.lower().startswith(name.lower().replace("-", "_") + "-")]
            ok = [w for w in from_links if wheel_matches(w, args.python_version, platforms, arch)]
            if ok:
                print(f"  OK     {name:24} {ver:12} {ok[0]}  (find-links)")
            elif from_links:
                bad += 1
                print(f"  NO_WHEEL {name:24} {ver:12} in find-links but not for the target: {from_links[0]}")
            else:
                bad += 1
                print(f"  MISSING  {name:24} {ver:12} neither on PyPI nor in find-links")
            continue
        status, detail, _ = classify(files, args.python_version, platforms, arch)
        if status == "OK":
            print(f"  OK     {name:24} {ver:12} {detail}")
        elif status == "SDIST":
            warn += 1
            print(f"  BUILD  {name:24} {ver:12} no wheel for the target — compiles from sdist ({detail})")
        else:
            bad += 1
            print(f"  {status:6} {name:24} {ver:12} no wheel for cp{args.python_version}/{arch}: available tags: {detail}")

    print(f"\n{len(reqs) - bad - warn} ok, {warn} sdist-only (warning), {bad} would fail the build")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
