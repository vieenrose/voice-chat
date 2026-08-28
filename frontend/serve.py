#!/usr/bin/env python3
"""Static server for the voice-chat frontend with correct cache headers.

- index.html: no-store (always fresh -> picks up new hashed JS builds)
- /assets/*: immutable max-age=1y (content-addressed by vite hash)
- others: no-cache
Usage: python3 serve.py [port] [--directory DIR]
"""
import argparse
import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        path = self.path.split("?", 1)[0]
        if path.endswith("index.html") or path == "/":
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        elif path.startswith("/assets/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=5173)
    ap.add_argument("--directory", default=None)
    args, remaining = ap.parse_known_args()
    handler = functools.partial(CacheHandler, directory=args.directory)
    with ThreadingHTTPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"serving {args.directory or os.getcwd()} on :{args.port} (cache-aware)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()