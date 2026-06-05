using System.IO.Compression;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;

internal static class ReferenceStore
{
    // Flat row-major layout: vector i starts at i * STRIDE (physical)
    // Dims=14 logical dimensions; STRIDE=16 adds 2 zero-padded slots for Vector256<short> AVX2 MADD.
    // short (int16, scale Q=4096): 2 bytes/value → 96 MB vs 192 MB float per instance
    // Sentinel -1.0 → -Q exactly; values [0,1] → [0,Q]. Max |diff|=2Q=8192 ≤ 32767 (no int16 overflow).
    internal static short[] Vectors = [];
    internal static bool[] Labels = [];
    internal static int Count;

    // IVF index — written once at startup, read-only during queries.
    // Centroids stored as float32 (not Half): 2000×16×4 = 128 KB; full precision for cluster boundaries.
    // STRIDE_FLOAT=16: 14 real dims + 2 zero-padded for Vector256<float> AVX2 alignment.
    internal static float[] Centroids = [];
    internal static int[] InvertedIndex = [];  // flat, Count entries ordered by cluster
    internal static int[] ClusterStart = [];   // ClusterStart[c] = first position of cluster c in InvertedIndex
    internal static int[] ClusterSize = [];    // number of vectors in cluster c

    internal const int K_CLUSTERS = 4000;
    internal const int NPROBE = 30;
    private const int BatchSize = 15_000;
    private const int NIterations = 200;
    private const int Dims = 14;
    internal const int STRIDE = 16; // short[] physical stride: 14 real dims + 2 zero-padding for AVX2 MADD
    internal const int STRIDE_FLOAT = 16; // float[] centroid stride: 14 real dims + 2 zero-padding for AVX2 float
    internal const int Q = 4096; // quantization scale: float v → (short)(v * Q)

    /// <summary>
    /// Runtime path: try to load the pre-built IVF binary index first.
    /// If not found, fall back to the full JSON+gzip+K-Means pipeline (for local dev).
    /// </summary>
    internal static async Task LoadAsync(string resourcesPath)
    {
        var indexPath = Path.Combine(resourcesPath, "ivf_index.bin");
        if (File.Exists(indexPath))
        {
            LoadBinary(indexPath);
            Console.WriteLine($"[ReferenceStore] IVF index loaded ({Count:N0} vectors, K={K_CLUSTERS})");
            return;
        }

        // Fallback: full pipeline for local development without a pre-built index.
        await LoadReferencesJson(resourcesPath);
        BuildIvfIndex();
    }

    /// <summary>
    /// --build-index path: load references.json.gz, run K-Means + cluster sort,
    /// serialize to ivf_index.bin. Called once during `docker build`.
    /// The index-builder stage has no memory limit, so the peak ~145 MB
    /// during K-Means is safe. At runtime, LoadBinary uses only ~100 MB.
    /// </summary>
    internal static async Task BuildIndexAsync(string resourcesPath)
    {
        await LoadReferencesJson(resourcesPath);
        BuildIvfIndex();
        var indexPath = Path.Combine(resourcesPath, "ivf_index.bin");
        SaveBinary(indexPath);
        Console.WriteLine($"[ReferenceStore] IVF index saved to {indexPath}");
    }

    private static async Task LoadReferencesJson(string resourcesPath)
    {
        const int MaxVectors = 3_000_000;

        var vectors = new short[MaxVectors * STRIDE];
        var labels = new bool[MaxVectors];

        var path = Path.Combine(resourcesPath, "references.json.gz");
        await using var fileStream = File.OpenRead(path);
        await using var gzip = new GZipStream(fileStream, CompressionMode.Decompress);

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        int count = 0;

        await foreach (var entry in JsonSerializer.DeserializeAsyncEnumerable<ReferenceEntry>(gzip, options))
        {
            if (entry is null) continue;

            int offset = count * STRIDE;
            var v = entry.Vector;
            for (int d = 0; d < Dims; d++)
                vectors[offset + d] = (short)(v[d] * Q);

            labels[count] = entry.Label == "fraud";
            count++;
        }

        Vectors = vectors;
        Labels = labels;
        Count = count;

        Console.WriteLine($"[ReferenceStore] Loaded {count:N0} vectors");
    }

    private static void BuildIvfIndex()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rng = new Random(42);

        var centroids = new float[K_CLUSTERS * STRIDE_FLOAT];

        // Phase A — KMeans++ initialization.
        // Each of the K centroids is selected sequentially with probability proportional
        // to its squared distance from the nearest already-chosen centroid (D² sampling).
        // We estimate D² over a small batch (KPP_BATCH=1000) rather than the full 3M set,
        // keeping total cost O(K × KPP_BATCH × K/2 × Dims) ≈ 56B float ops ≈ 30s with AVX2.
        // This is a build-time-only cost; runtime is completely unaffected.
        // Quality: D² sampling on 1000 random vectors already spreads centroids far better
        // than pure random init, dramatically reducing dead centroids and imbalanced clusters.
        const int KppBatch = 1000;
        var kppIndices = new int[KppBatch];
        Span<float> kppBuf = stackalloc float[STRIDE_FLOAT]; // zero-padded to 16 for AVX2
        var kppDistSq = new float[KppBatch];

        // 1st centroid: uniform random
        int firstIdx = rng.Next(Count);
        int firstOff = firstIdx * STRIDE;
        for (int d = 0; d < Dims; d++)
            centroids[d] = Vectors[firstOff + d] / (float)Q;

        for (int c = 1; c < K_CLUSTERS; c++)
        {
            // Sample KppBatch random vectors
            for (int b = 0; b < KppBatch; b++)
                kppIndices[b] = rng.Next(Count);

            float totalDistSq = 0f;

            for (int b = 0; b < KppBatch; b++)
            {
                // Dequantize to float and zero-pad for AVX2
                int srcOff = kppIndices[b] * STRIDE;
                for (int d2 = 0; d2 < Dims; d2++)
                    kppBuf[d2] = Vectors[srcOff + d2] / (float)Q;

                // Find nearest among the c already-chosen centroids (AVX2 path)
                float best = float.MaxValue;
                if (Avx2.IsSupported)
                {
                    var qLo = Vector256.LoadUnsafe(ref kppBuf[0]);
                    var qHi = Vector256.LoadUnsafe(ref kppBuf[8]);
                    for (int j = 0; j < c; j++)
                    {
                        int jOff = j * STRIDE_FLOAT;
                        var cLo = Vector256.LoadUnsafe(ref centroids[jOff]);
                        var cHi = Vector256.LoadUnsafe(ref centroids[jOff + 8]);
                        var dLo = Avx.Subtract(qLo, cLo);
                        var dHi = Avx.Subtract(qHi, cHi);
                        var sqLo = Avx.Multiply(dLo, dLo);
                        var sqHi = Avx.Multiply(dHi, dHi);
                        var sum8 = Avx.Add(sqLo, sqHi);
                        var lo4 = sum8.GetLower();
                        var hi4 = sum8.GetUpper();
                        var sum4 = Sse.Add(lo4, hi4);
                        var ha1 = Sse3.HorizontalAdd(sum4, sum4);
                        var ha2 = Sse3.HorizontalAdd(ha1, ha1);
                        float dist = ha2[0];
                        if (dist < best) best = dist;
                    }
                }
                else
                {
                    for (int j = 0; j < c; j++)
                    {
                        float dist = 0f;
                        int jOff = j * STRIDE_FLOAT;
                        for (int d2 = 0; d2 < Dims; d2++)
                        {
                            float diff = kppBuf[d2] - centroids[jOff + d2];
                            dist += diff * diff;
                        }
                        if (dist < best) best = dist;
                    }
                }
                kppDistSq[b] = best;
                totalDistSq += best;
            }

            // Roulette-wheel: pick a sample with probability ∝ D²
            float target = (float)rng.NextDouble() * totalDistSq;
            float cumulative = 0f;
            int chosenB = KppBatch - 1;
            for (int b = 0; b < KppBatch; b++)
            {
                cumulative += kppDistSq[b];
                if (cumulative >= target) { chosenB = b; break; }
            }

            int chosenIdx = kppIndices[chosenB];
            int dstOff = c * STRIDE_FLOAT;
            int chosenOff = chosenIdx * STRIDE;
            for (int d = 0; d < Dims; d++)
                centroids[dstOff + d] = Vectors[chosenOff + d] / (float)Q;
        }

        Console.WriteLine($"[IVF] KMeans++ init done in {sw.ElapsedMilliseconds}ms (K={K_CLUSTERS})");
        sw.Restart();

        // Phase B — Mini-batch K-Means (NIterations × BatchSize random vectors).
        // Online mean update: c[d] += (v[d] - c[d]) / ++count[c]
        // Numerically stable: avoids accumulating large float sums across the full dataset.
        var batchCounts = new int[K_CLUSTERS];
        var batchIndices = new int[BatchSize];
        Span<float> vBuf = stackalloc float[Dims];

        for (int iter = 0; iter < NIterations; iter++)
        {
            Array.Clear(batchCounts, 0, K_CLUSTERS);

            for (int b = 0; b < BatchSize; b++)
                batchIndices[b] = rng.Next(Count);

            for (int b = 0; b < BatchSize; b++)
            {
                int srcOff = batchIndices[b] * STRIDE;
                for (int d = 0; d < Dims; d++)
                    vBuf[d] = Vectors[srcOff + d] / (float)Q;

                int c = FindNearestCentroid(vBuf, centroids);
                int cnt = ++batchCounts[c];
                int cOff = c * STRIDE_FLOAT;
                float scale = 1f / cnt;
                for (int d = 0; d < Dims; d++)
                    centroids[cOff + d] += (vBuf[d] - centroids[cOff + d]) * scale;
            }

            // Dead centroid guard: a centroid that receives no updates in a batch drifts into
            // uselessness. Reinitialize it to a random data point so it stays competitive.
            for (int c = 0; c < K_CLUSTERS; c++)
            {
                if (batchCounts[c] == 0)
                {
                    int srcOff = rng.Next(Count) * STRIDE;
                    int dstOff = c * STRIDE_FLOAT;
                    for (int d = 0; d < Dims; d++)
                        centroids[dstOff + d] = Vectors[srcOff + d] / (float)Q;
                }
            }
        }

        Console.WriteLine($"[IVF] K-Means done in {sw.ElapsedMilliseconds}ms");
        sw.Restart();

        // Publish centroids before Phase C so FindNearestCentroid can use the static field.
        Centroids = centroids;

        // Phase C — Full assignment (parallel).
        // Parallel.For: each thread writes assignments[i] for its own i → no false sharing.
        // CA2014 suppressed: stackalloc is inside a delegate (separate stack frame per call),
        // not a raw loop — stack is freed on each lambda return, no accumulation.
        // ushort (not int) for cluster IDs: K_CLUSTERS=2000 fits in ushort (max 65535) → 6 MB vs 12 MB.
        var assignments = new ushort[Count];
#pragma warning disable CA2014
        Parallel.For(0, Count, i =>
        {
            int srcOff = i * STRIDE;
            Span<float> vBuf = stackalloc float[Dims];
            for (int d = 0; d < Dims; d++)
                vBuf[d] = Vectors[srcOff + d] / (float)Q;
            assignments[i] = (ushort)FindNearestCentroid(vBuf, Centroids);
        });
#pragma warning restore CA2014

        // Count cluster sizes (sequential scan — avoids Interlocked overhead on int[2000]).
        var clusterSize = new int[K_CLUSTERS];
        for (int i = 0; i < Count; i++)
            clusterSize[assignments[i]]++;

        // Prefix-sum: ClusterStart[c] = ∑ ClusterSize[0..c-1]
        var clusterStart = new int[K_CLUSTERS];
        for (int c = 1; c < K_CLUSTERS; c++)
            clusterStart[c] = clusterStart[c - 1] + clusterSize[c - 1];

        // Scatter: fill flat InvertedIndex using a write cursor per cluster.
        var invertedIndex = new int[Count];
        var cursor = new int[K_CLUSTERS];
        Array.Copy(clusterStart, cursor, K_CLUSTERS);
        for (int i = 0; i < Count; i++)
        {
            int c = assignments[i];
            invertedIndex[cursor[c]++] = i;
        }

        // Explicitly release assignments (6 MB) before Phase D allocates visited[] (3 MB).
        // Without this, the peak is: Vectors(96) + Labels(3) + assignments(6) + invertedIndex(12) + visited(3) + runtime ≈ 162 MB > 160 MB limit.
        // The JIT does not guarantee GC between these allocations in the same method frame.
        assignments = null!;
        GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);

        ClusterStart = clusterStart;
        ClusterSize = clusterSize;

        long avgCluster = Count / K_CLUSTERS;
        Console.WriteLine($"[IVF] Index ready in {sw.ElapsedMilliseconds}ms | K={K_CLUSTERS} avg_cluster={avgCluster:N0} nprobe={NPROBE}");
        sw.Restart();

        // Phase D — In-place cluster sort.
        // Reorder Vectors[] and Labels[] so sorted position s holds the vector
        // previously at invertedIndex[s]. Stage 2 in KnnSearch then scans
        // clusterStart[c]..clusterStart[c]+clusterSize[c]-1 sequentially,
        // engaging the hardware prefetcher and eliminating DRAM random-access latency.
        var visited = new bool[Count]; // 3 MB, freed after method exits
        Span<short> tempVec = stackalloc short[STRIDE]; // 32 bytes on stack

        for (int start = 0; start < Count; start++)
        {
            if (visited[start]) continue;

            int src = invertedIndex[start];
            if (src == start)
            {
                visited[start] = true;
                continue; // trivial 1-cycle: already in place
            }

            // Save the element at the head of this cycle.
            int startOff = start * STRIDE;
            for (int d = 0; d < STRIDE; d++) tempVec[d] = Vectors[startOff + d];
            bool tempLabel = Labels[start];

            int cur = start;
            while (true)
            {
                int nxt = invertedIndex[cur];
                visited[cur] = true;
                if (nxt == start) break;

                // Pull data from nxt into cur.
                int curOff = cur * STRIDE;
                int nxtOff = nxt * STRIDE;
                for (int d = 0; d < STRIDE; d++) Vectors[curOff + d] = Vectors[nxtOff + d];
                Labels[cur] = Labels[nxt];
                cur = nxt;
            }

            // Close the cycle: cur is the last node whose successor was start.
            int closeOff = cur * STRIDE;
            for (int d = 0; d < STRIDE; d++) Vectors[closeOff + d] = tempVec[d];
            Labels[cur] = tempLabel;
        }

        // InvertedIndex no longer needed after sort — release 12 MB.
        InvertedIndex = Array.Empty<int>();
        GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
        Console.WriteLine($"[IVF] Cluster sort done in {sw.ElapsedMilliseconds}ms");
    }

    // ---- Binary serialization ----

    /// <summary>
    /// Save the fully built IVF index to a binary file.
    /// Format: version(4) | count(4) | K_CLUSTERS(4) | NPROBE(4) | labels(count bytes)
    /// | vectors(count*STRIDE*2 bytes as short) | centroids(K*STRIDE_FLOAT*4 bytes)
    /// | clusterStart(K*4 bytes) | clusterSize(K*4 bytes)
    ///
    /// Total: ~12 + 3 + 96 + 0.13 + 0.008 + 0.008 ≈ 99 MB
    /// </summary>
    private static void SaveBinary(string path)
    {
        using var fs = File.OpenWrite(path);
        using var bw = new BinaryWriter(fs);

        bw.Write(1); // version
        bw.Write(Count);
        bw.Write(K_CLUSTERS);
        bw.Write(NPROBE);

        // Labels: bool → byte
        for (int i = 0; i < Count; i++)
            bw.Write(Labels[i]);

        // Vectors: short[]
        for (int i = 0; i < Count * STRIDE; i++)
            bw.Write(Vectors[i]);

        // Centroids: float[]
        for (int i = 0; i < K_CLUSTERS * STRIDE_FLOAT; i++)
            bw.Write(Centroids[i]);

        // ClusterStart: int[]
        for (int i = 0; i < K_CLUSTERS; i++)
            bw.Write(ClusterStart[i]);

        // ClusterSize: int[]
        for (int i = 0; i < K_CLUSTERS; i++)
            bw.Write(ClusterSize[i]);

        Console.WriteLine($"[ReferenceStore] Saved IVF index: {new FileInfo(path).Length / 1_000_000} MB");
    }

    /// <summary>
    /// Load a pre-built IVF index from disk. Fast path for runtime startup (~2-3s).
    /// </summary>
    private static void LoadBinary(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        int version = br.ReadInt32();
        if (version != 1)
            throw new InvalidDataException($"IVF index version mismatch: expected 1, got {version}");

        Count = br.ReadInt32();
        int kClusters = br.ReadInt32();
        int nprobe = br.ReadInt32();

        // Validate constants match what was baked at build time
        if (kClusters != K_CLUSTERS || nprobe != NPROBE)
            throw new InvalidDataException(
                $"IVF params mismatch: built(K={kClusters},nprobe={nprobe}) vs code(K={K_CLUSTERS},nprobe={NPROBE})");

        Labels = new bool[Count];
        Vectors = new short[Count * STRIDE];
        Centroids = new float[K_CLUSTERS * STRIDE_FLOAT];
        ClusterStart = new int[K_CLUSTERS];
        ClusterSize = new int[K_CLUSTERS];

        for (int i = 0; i < Count; i++)
            Labels[i] = br.ReadBoolean();

        for (int i = 0; i < Count * STRIDE; i++)
            Vectors[i] = br.ReadInt16();

        for (int i = 0; i < K_CLUSTERS * STRIDE_FLOAT; i++)
            Centroids[i] = br.ReadSingle();

        for (int i = 0; i < K_CLUSTERS; i++)
            ClusterStart[i] = br.ReadInt32();

        for (int i = 0; i < K_CLUSTERS; i++)
            ClusterSize[i] = br.ReadInt32();
    }

    // Finds the nearest centroid by squared Euclidean distance.
    // Used during K-Means training (local centroids) and Phase C full assignment (static Centroids).
    // AVX2 path: loads 2×float8 per centroid (STRIDE_FLOAT=16; dims 14-15 are zero-padded in both
    // query and centroids, so their contribution to the squared distance is always zero).
    private static int FindNearestCentroid(ReadOnlySpan<float> vBuf, float[] centroids)
    {
        float bestDist = float.MaxValue;
        int bestC = 0;

        if (Avx2.IsSupported)
        {
            // Zero-pad query to STRIDE_FLOAT so SIMD reads 16 floats safely.
            Span<float> qPad = stackalloc float[STRIDE_FLOAT]; // zero-initialized; dims 14-15 stay 0
            vBuf.CopyTo(qPad);
            var qLo = Vector256.LoadUnsafe(ref qPad[0]);  // dims 0-7
            var qHi = Vector256.LoadUnsafe(ref qPad[8]);  // dims 8-15

            for (int c = 0; c < K_CLUSTERS; c++)
            {
                int cOff = c * STRIDE_FLOAT;
                var cLo = Vector256.LoadUnsafe(ref centroids[cOff]);
                var cHi = Vector256.LoadUnsafe(ref centroids[cOff + 8]);
                var dLo = Avx.Subtract(qLo, cLo);
                var dHi = Avx.Subtract(qHi, cHi);
                var sqLo = Avx.Multiply(dLo, dLo);
                var sqHi = Avx.Multiply(dHi, dHi);
                // Pairwise add: sum8[i] = sqLo[i] + sqHi[i] (pairs dims 0+8, 1+9, ..., 7+15)
                var sum8 = Avx.Add(sqLo, sqHi);
                var lo4 = sum8.GetLower();
                var hi4 = sum8.GetUpper();
                var sum4 = Sse.Add(lo4, hi4);
                var ha1 = Sse3.HorizontalAdd(sum4, sum4);
                var ha2 = Sse3.HorizontalAdd(ha1, ha1);
                float dist = ha2[0];
                if (dist < bestDist) { bestDist = dist; bestC = c; }
            }
        }
        else
        {
            for (int c = 0; c < K_CLUSTERS; c++)
            {
                int cOff = c * STRIDE_FLOAT;
                float dist = 0f;
                for (int d = 0; d < Dims; d++)
                {
                    float diff = vBuf[d] - centroids[cOff + d];
                    dist += diff * diff;
                }
                if (dist < bestDist) { bestDist = dist; bestC = c; }
            }
        }
        return bestC;
    }

    private sealed class ReferenceEntry
    {
        public float[] Vector { get; set; } = [];
        public string Label { get; set; } = "";
    }
}
