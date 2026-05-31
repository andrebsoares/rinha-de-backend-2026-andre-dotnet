using System.IO.Compression;
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
    // Centroids stored as float32 (not Half): 200×14×4 = 11 KB; full precision for cluster boundaries.
    internal static float[] Centroids = [];
    internal static int[] InvertedIndex = [];  // flat, Count entries ordered by cluster
    internal static int[] ClusterStart = [];   // ClusterStart[c] = first position of cluster c in InvertedIndex
    internal static int[] ClusterSize = [];    // number of vectors in cluster c

    internal const int K_CLUSTERS = 200;
    internal const int NPROBE = 10;
    private const int BatchSize = 10_000;
    private const int NIterations = 25;
    private const int Dims = 14;
    internal const int STRIDE = 16; // physical stride: 14 real dims + 2 zero-padding for SIMD alignment
    internal const int Q = 4096; // quantization scale: float v → (short)(v * Q)

    internal static async Task LoadAsync(string resourcesPath)
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

        BuildIvfIndex();
    }

    private static void BuildIvfIndex()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rng = new Random(42);

        // Phase A — Random initialization.
        // Pick K_CLUSTERS distinct vector indices as starting centroids.
        // Simple random init keeps startup time predictable (~0ms).
        // K-Means++ would improve initial quality but costs K extra full-data passes (~6s extra).
        var centroids = new float[K_CLUSTERS * Dims];
        var chosen = new HashSet<int>(K_CLUSTERS);
        for (int c = 0; c < K_CLUSTERS; c++)
        {
            int idx;
            do { idx = rng.Next(Count); } while (!chosen.Add(idx));
            int srcOff = idx * STRIDE;
            int dstOff = c * Dims;
            for (int d = 0; d < Dims; d++)
                centroids[dstOff + d] = Vectors[srcOff + d] / (float)Q;
        }

        // Phase B — Mini-batch K-Means (NIterations × BatchSize random vectors).
        // Online mean update: c[d] += (v[d] - c[d]) / ++count[c]
        // Numerically stable: avoids accumulating large float sums across the full dataset.
        var batchCounts = new int[K_CLUSTERS];
        var batchIndices = new int[BatchSize];
        // Declared outside the loop: stackalloc inside a loop grows the stack frame each iteration.
        // 25 iterations × 14 floats × 4 bytes = 1.4 KB growth — technically safe but triggers CA2014.
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
                int cOff = c * Dims;
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
                    int dstOff = c * Dims;
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
        // byte (not int) for cluster IDs: K_CLUSTERS=200 fits in byte (max 255) → 3 MB vs 12 MB.
        var assignments = new byte[Count];
#pragma warning disable CA2014
        Parallel.For(0, Count, i =>
        {
            int srcOff = i * STRIDE;
            Span<float> vBuf = stackalloc float[Dims];
            for (int d = 0; d < Dims; d++)
                vBuf[d] = Vectors[srcOff + d] / (float)Q;
            assignments[i] = (byte)FindNearestCentroid(vBuf, Centroids);
        });
#pragma warning restore CA2014

        // Count cluster sizes (sequential scan — avoids Interlocked overhead on int[200]).
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

        // Explicitly release assignments (12 MB) before Phase D allocates visited[] (3 MB).
        // Without this, the peak is: Vectors(96) + Labels(3) + assignments(12) + invertedIndex(12) + visited(3) + runtime ≈ 161 MB > 160 MB limit.
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

    // Finds the nearest centroid by squared Euclidean distance.
    // Used during K-Means training (local centroids array) and full assignment (static Centroids).
    private static int FindNearestCentroid(ReadOnlySpan<float> vBuf, float[] centroids)
    {
        float bestDist = float.MaxValue;
        int bestC = 0;
        for (int c = 0; c < K_CLUSTERS; c++)
        {
            int cOff = c * Dims;
            float dist = 0f;
            for (int d = 0; d < Dims; d++)
            {
                float diff = vBuf[d] - centroids[cOff + d];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; bestC = c; }
        }
        return bestC;
    }

    private sealed class ReferenceEntry
    {
        public float[] Vector { get; set; } = [];
        public string Label { get; set; } = "";
    }
}
