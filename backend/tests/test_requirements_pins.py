"""
Tests for backend/check_requirements.py — the pure parts, offline.

These exist because the regenerated requirements.txt shipped one pin that matched nothing
on PyPI (`qwen-agent==0.34.*`; the project versions itself 0.0.34), which `pip install`
resolves as an immediate error inside `docker build`. Nothing else in the repo would have
noticed until the build failed.
"""
import importlib.util
import os
import sys
import unittest
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BACKEND)

# Loaded by path: the module name would otherwise shadow nothing, but the file also has
# a __main__ block and lives outside the package.
spec = importlib.util.spec_from_file_location("check_requirements", os.path.join(BACKEND, "check_requirements.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


class TestParseRequirements(unittest.TestCase):
    def test_comments_extras_markers_and_find_links(self):
        text = "\n".join([
            "# a comment",
            "",
            "--find-links https://huggingface.co/datasets/andito/x/tree/main/whl/cu124",
            "fastapi==0.141.*",
            "uvicorn[standard]==0.52.*",
            "onnxruntime==1.29.* ; sys_platform == 'darwin'",
            "qwentts-cpp-python==0.3.1+cu124",
            "lxml==6.1.*                 # inline comment",
        ])
        reqs, links = cr.parse_requirements(text)
        self.assertEqual([r["name"] for r in reqs],
                         ["fastapi", "uvicorn", "onnxruntime", "qwentts-cpp-python", "lxml"])
        self.assertEqual(links, ["https://huggingface.co/datasets/andito/x/tree/main/whl/cu124"])
        by = {r["name"]: r for r in reqs}
        self.assertEqual(by["uvicorn"]["extras"], "standard")
        self.assertEqual(by["onnxruntime"]["marker"], "sys_platform == 'darwin'")
        self.assertEqual(by["qwentts-cpp-python"]["spec"], "==0.3.1+cu124")
        self.assertEqual(by["lxml"]["spec"], "==6.1.*")          # trailing comment stripped

    def test_real_requirements_file_parses(self):
        with open(os.path.join(BACKEND, "requirements.txt"), encoding="utf-8") as f:
            reqs, links = cr.parse_requirements(f.read())
        self.assertGreater(len(reqs), 20)
        self.assertTrue(links, "the TTS CUDA wheel index must stay declared")


class TestPickVersion(unittest.TestCase):
    RELEASES = ["0.0.32", "0.0.34", "0.1.0", "0.2.0", "1.0.0", "1.0.1", "1.1.0",
                "2.0.0rc1", "2.0.0", "2.1.0", "3.0.0.post1"]

    def test_prefix_pin_against_wrong_major_guess(self):
        # The shipped bug: qwen-agent versions as 0.0.NN. `==0.34.*` matches nothing and
        # pip fails the build with "No matching distribution found".
        self.assertIsNone(cr.pick_version("==0.34.*", self.RELEASES))
        self.assertEqual(cr.pick_version("==0.0.34.*", self.RELEASES), "0.0.34")

    def test_prefix_and_ge_and_bare(self):
        self.assertEqual(cr.pick_version("==1.*", self.RELEASES), "1.1.0")
        self.assertEqual(cr.pick_version(">=2.0", self.RELEASES), "3.0.0.post1")
        self.assertEqual(cr.pick_version("", self.RELEASES), "3.0.0.post1")

    def test_prerelease_and_local_versions_are_not_candidates(self):
        self.assertEqual(cr.pick_version("==2.*", self.RELEASES), "2.1.0")
        self.assertIsNone(cr.pick_version("==9.*", []))


class TestWheelMatches(unittest.TestCase):
    PLAT = ("manylinux", "musllinux")

    def m(self, fn, arch="x86_64", pyver="313"):
        return cr.wheel_matches(fn, pyver, self.PLAT, arch)

    def test_pure_python_and_exact_abi(self):
        self.assertTrue(self.m("fastapi-0.141.1-py3-none-any.whl"))
        self.assertTrue(self.m("numpy-2.4.6-cp313-cp313-manylinux_2_28_x86_64.whl"))

    def test_stable_abi_is_forward_compatible(self):
        # psutil ships cp36-abi3-manylinux wheels; that is how cp313 gets psutil.
        self.assertTrue(self.m("psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.whl"))
        self.assertTrue(self.m("psutil-7.2.2-cp38-abi3-manylinux2010_x86_64.whl"))
        self.assertFalse(self.m("psutil-7.2.2-cp314-abi3-manylinux2010_x86_64.whl"),
                         "a wheel built for a NEWER interpreter cannot be installed")

    def test_architecture_is_not_ignored(self):
        # The TTS CUDA wheel index ships aarch64 AND x86_64 side by side; a checker that
        # only substring-matched "manylinux" would green-light an x86_64 build that fails.
        self.assertTrue(self.m("qwentts_cpp_python-0.3.1+cu124-py3-none-manylinux_2_35_x86_64.whl"))
        self.assertFalse(self.m("qwentts_cpp_python-0.3.1+cu124-py3-none-manylinux_2_35_aarch64.whl"))
        self.assertTrue(self.m("qwentts_cpp_python-0.3.1+cu124-py3-none-manylinux_2_35_aarch64.whl", arch="aarch64"))

    def test_other_interpreter_os_and_freethreaded_are_rejected(self):
        self.assertFalse(self.m("thing-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"),
                         "non-abi3 cp312 wheels are not installable on cp313")
        self.assertFalse(self.m("thing-1.0-cp313-cp313-macosx_11_0_arm64.whl"))
        self.assertFalse(self.m("thing-1.0-cp313-cp313t-manylinux_2_28_x86_64.whl"),
                         "free-threaded (cp313t) builds do not satisfy a normal cp313")
        self.assertFalse(self.m("thing-1.0.tar.gz"))


class TestClassify(unittest.TestCase):
    def test_ok_then_no_wheel_then_sdist(self):
        files = [{"filename": "x-1.0-py3-none-any.whl"}]
        self.assertEqual(cr.classify(files, "313", ("manylinux",))[0], "OK")
        files = [{"filename": "x-1.0-cp311-cp311-manylinux_2_28_x86_64.whl"}]
        self.assertEqual(cr.classify(files, "313", ("manylinux",))[0], "NO_WHEEL")
        files = [{"filename": "jieba-0.42.1.tar.gz"}]
        self.assertEqual(cr.classify(files, "313", ("manylinux",))[0], "SDIST")


class TestPinsMatchThisEnvironment(unittest.TestCase):
    """Each pin must match the version the demo is actually running, or the Dockerfile
    installs a different (and probably untested) stack than the one validated here."""

    def test_installed_versions_satisfy_the_pins(self):
        with open(os.path.join(BACKEND, "requirements.txt"), encoding="utf-8") as f:
            reqs, _ = cr.parse_requirements(f.read())
        checked = 0
        for r in reqs:
            if r["marker"] and "darwin" in r["marker"]:
                continue
            try:
                installed = metadata.version(r["name"])
            except metadata.PackageNotFoundError:
                continue                      # fallback adapter not installed here
            cand = cr.pick_version(r["spec"], [installed])
            self.assertEqual(
                cand, installed,
                f"pin {r['raw']!r} does not match the installed {r['name']}=={installed} "
                f"— the Docker image would build a different stack than the one tested")
            checked += 1
        self.assertGreaterEqual(checked, 20, f"only {checked} pins were checkable — too weak to mean anything")


if __name__ == "__main__":
    unittest.main(verbosity=2)
