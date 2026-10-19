#!/usr/bin/env python3
"""Turn PULSAR's one-shot fwd35 into a persistent request server.

The original binary loads the GGUF, runs one prompt and exits, so every turn would pay the full
weight load. This script derives fwd35_serve.cu from fwd35_sm53.cu: the model is loaded once, then
each stdin line is a request `N id id ...` answered with the greedy ids one per line and `END`.
Recurrent state (GDN conv/ssm) is zeroed per request; the K/V cache is position-indexed and the
attention only reads positions below the current one, so stale entries are never read.

A request `+N id id ...` instead continues from the state the previous request left (K/V cache and
recurrent state up to its last fed token): a chat turn then prefills only its new tokens. Every
request ends with `P n` (tokens now held in the state) before `END`; app/llm.py uses it to decide
whether the next prompt can be sent as a continuation.

A reader thread owns stdin. A line `STOP` sets a flag that ends the decode of the request in
progress after the current token (barge-in); it is never queued as a request.

usage: make_serve.py fwd35_sm53.cu fwd35_serve.cu
"""
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()


def sub(old, new, count=1):
    global src
    if src.count(old) != count:
        raise SystemExit("patch anchor found %d times, expected %d: %r" % (src.count(old), count, old[:60]))
    src = src.replace(old, new)


src = ("#include <vector>\n#include <cstring>\n#include <cstdlib>\n#include <string>\n#include <deque>\n"
       "#include <mutex>\n#include <condition_variable>\n#include <atomic>\n#include <thread>\n") + src
sub('    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf N id id ...\\n", argv[0]); return 1; }',
    '    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf  (requests on stdin: N id id ...)\\n", argv[0]); return 1; }')
sub("    int N = atoi(argv[2]); const int np = argc - 3;", "    const int np = 0; int N = 0;")
sub("    if (np + N > CTX) { fprintf(stderr, \"prompt + generation exceeds ctx %d\\n\", CTX); return 1; }\n", "")
sub("    int *hpos; CK(cudaMallocHost(&hpos, sizeof(int) * (np + N + 1)));",
    "    int *hpos; CK(cudaMallocHost(&hpos, sizeof(int) * (CTX + 1)));")
sub("    const int tiny = np >= 2;", "    const int tiny = 1; /* graph warm-up path; per request below */")

head, sep, _ = src.partition("    int pos = 0;\n    tcur = 0;\n")
if not sep:
    raise SystemExit("serve anchor missing")

tail = r'''    cudaEvent_t evx[2];
    for (int k = 0; k < 2; k++) CK(cudaEventCreateWithFlags(&evx[k], cudaEventDisableTiming));
    fprintf(stderr, "fwd35: ready, serving requests on stdin\n");

    /* stdin is owned by one reader thread. Each request gets a sequence number; a STOP line cuts the
       latest request sent so far, whether it is still queued, prefilling or decoding. */
    std::atomic<int> stop_upto(0);
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<std::pair<int, std::string>> queue;
    bool eof = false;
    std::thread reader([&]() {
        char *l = NULL; size_t c = 0; ssize_t n;
        int seq = 0;
        while ((n = getline(&l, &c, stdin)) > 0) {
            if (strncmp(l, "STOP", 4) == 0) { stop_upto.store(seq); continue; }
            ++seq;
            {
                std::lock_guard<std::mutex> lk(qm);
                queue.push_back(std::make_pair(seq, std::string(l, (size_t)n)));
            }
            qcv.notify_one();
        }
        free(l);
        {
            std::lock_guard<std::mutex> lk(qm);
            eof = true;
        }
        qcv.notify_one();
    });

    std::vector<int> ids;
    /* Positions already in the K/V cache and recurrent states. A request starting with "+"
       continues from there (multi-turn chat: only the new tokens are prefilled); any other request
       starts from zero. Every request ends with "P <n>" (tokens now in the state) before END. */
    int kpos = 0;
    for (;;) {
        std::string req; int my_seq = 0;
        {
            std::unique_lock<std::mutex> lk(qm);
            qcv.wait(lk, [&]() { return !queue.empty() || eof; });
            if (queue.empty()) break;
            my_seq = queue.front().first;
            req = std::move(queue.front().second);
            queue.pop_front();
        }
        char *p = &req[0], *e = NULL;
        while (*p == ' ') p++;
        const int cont = *p == '+';
        if (cont) p++;
        const long nreq = strtol(p, &e, 10);
        if (e == p || nreq < 1) { printf("ERR\nEND\n"); fflush(stdout); continue; }
        p = e; ids.clear();
        for (;;) { const long v = strtol(p, &e, 10); if (e == p) break; ids.push_back((int)v); p = e; }
        const int npr = (int)ids.size(), N2 = (int)nreq;
        const int base = cont ? kpos : 0;
        if (npr < 1 || base + npr + N2 > CTX) { printf("ERR\nEND\n"); fflush(stdout); continue; }

        if (!cont) {
            for (int l = 0; l < NL; l++) if (!Ls[l].full) {
                CK(cudaMemsetAsync(d_conv[l], 0, 4 * (size_t)3 * CH, st));
                CK(cudaMemsetAsync(d_ssm[l], 0, 4 * (size_t)NV * DS * DS, st));
            }
            kpos = 0;
        }
        if (stop_upto.load() >= my_seq) { printf("P %d\nEND\n", kpos); fflush(stdout); continue; }
        const int tinyr = npr >= 2;
        int pos = base, chunk = 0;
        tcur = 0;
        const double t0 = now();
        for (int i = 0; i < npr;) {
            const int nt = npr - i < pch ? npr - i : pch;
            float *hb = hx + (size_t)E * NTM * (chunk & 1);
            if (chunk >= 2) CK(cudaEventSynchronize(evx[chunk & 1]));
            for (int t = 0; t < nt; t++) embed(ids[i + t], hb + (size_t)E * t);
            hpos[pos] = pos;
            CK(cudaMemcpyAsync(d_x, hb, 4 * (size_t)E * nt, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d_pos, hpos + pos, 4, cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(evx[chunk & 1], st));
            i += nt; pos += nt; chunk++;
            run(nt, tinyr, i == npr ? 1 : 0);
        }
        CK(cudaMemcpyAsync(hid, d_out, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const double t1 = now();
        int produced = 0;
        for (int s = 0; s < N2; s++) {
            const int a = *hid;
            printf("%d\n", a); fflush(stdout); produced++;
            tcur = s + 1;
            if (a == EOS_ID) break;
            if (s + 1 < N2) {
                if (stop_upto.load() >= my_seq) break;
                embed(a, hx);
                hpos[pos] = pos;
                CK(cudaMemcpyAsync(d_x, hx, 4 * (size_t)E, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(d_pos, hpos + pos, 4, cudaMemcpyHostToDevice, st));
                pos++;
                run(1, 0, 1);
                CK(cudaMemcpyAsync(hid, d_out, 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
            }
        }
        const double t2 = now();
        kpos = pos;  /* the last printed id is never fed back: it is the next request's first token */
        printf("P %d\nEND\n", kpos); fflush(stdout);
        CK(cudaGetLastError());
        fprintf(stderr, "fwd35: %s prefill %d tok %.3f s | decode %d tok %.3f s = %.2f tok/s%s\n",
                cont ? "cont" : "full", npr, t1 - t0, produced - 1, t2 - t1, produced > 1 ? (produced - 1) / (t2 - t1) : 0.0,
                stop_upto.load() >= my_seq ? " (stopped)" : "");
    }
    reader.join();
    return 0;
}
'''
open(dst_path, "w").write(head + tail)
print("wrote", dst_path)
