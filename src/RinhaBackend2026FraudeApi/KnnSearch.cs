using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

internal static class KnnSearch
{
    private const int K = 5;
    private const int Dims = 14;
    private const int STRIDE = ReferenceStore.STRIDE;
    private const int Q = ReferenceStore.Q;
    // Compile-time aliases — stay in sync with ReferenceStore without a runtime read.
    private const int NPROBE = ReferenceStore.NPROBE;

    internal static (float fraudScore, bool approved) Search(ReadOnlySpan<float> query)
    {
        // Stage 1: find the NPROBE nearest centroids.
        // K_CLUSTERS=2000 centroids × 14 dims via AVX2 (~0.5µs per centroid → ~1ms for 2000).
        // stackalloc: zero heap allocation, no GC pressure per request.
        Span<float> centDist = stackalloc float[NPROBE];
        Span<int> centId = stackalloc int[NPROBE];
        centDist.Fill(float.MaxValue);
        centId.Fill(-1);

        var centroids = ReferenceStore.Centroids;
        int kClusters = ReferenceStore.K_CLUSTERS;
        int strideFlt = ReferenceStore.STRIDE_FLOAT;

        // AVX2 path: 2×float8 loads per centroid (STRIDE_FLOAT=16; dims 14-15 zero-padded → zero contribution).
        if (Avx2.IsSupported)
        {
            Span<float> qPad = stackalloc float[ReferenceStore.STRIDE_FLOAT]; // zero-initialized
            for (int d = 0; d < Dims; d++) qPad[d] = query[d];
            var qLo = Vector256.LoadUnsafe(ref qPad[0]);
            var qHi = Vector256.LoadUnsafe(ref qPad[8]);

            for (int c = 0; c < kClusters; c++)
            {
                int cOff = c * strideFlt;
                var cLo = Vector256.LoadUnsafe(ref centroids[cOff]);
                var cHi = Vector256.LoadUnsafe(ref centroids[cOff + 8]);
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

                // Linear max-scan to find the eviction slot (NPROBE=20 → 19 compares; beats heap).
                int maxIdx = 0;
                for (int j = 1; j < NPROBE; j++)
                    if (centDist[j] > centDist[maxIdx]) maxIdx = j;
                if (dist < centDist[maxIdx]) { centDist[maxIdx] = dist; centId[maxIdx] = c; }
            }
        }
        else
        {
            for (int c = 0; c < kClusters; c++)
            {
                int cOff = c * strideFlt;
                float dist = 0f;
                for (int d = 0; d < Dims; d++)
                {
                    float diff = query[d] - centroids[cOff + d];
                    dist += diff * diff;
                }
                int maxIdx = 0;
                for (int j = 1; j < NPROBE; j++)
                    if (centDist[j] > centDist[maxIdx]) maxIdx = j;
                if (dist < centDist[maxIdx]) { centDist[maxIdx] = dist; centId[maxIdx] = c; }
            }
        }

        // Stage 2: scan vectors in the NPROBE selected clusters.
        // K=2000, avg_cluster=1500, NPROBE=20 → ~30,000 vectors vs 3M brute-force (~100× speedup).
        // Distance is computed in quantized int32 space (no float conversion in hot path).
        // Avx2.MultiplyAddAdjacent: 16 int16 diffs → 8 int32 pair-sums in 1 cycle → ~4× scalar speedup.
        Span<int> topDist = stackalloc int[K];
        Span<bool> topLabel = stackalloc bool[K];
        topDist.Fill(int.MaxValue);

        var vectors = ReferenceStore.Vectors;
        var labels = ReferenceStore.Labels;
        var clusterStart = ReferenceStore.ClusterStart;
        var clusterSize = ReferenceStore.ClusterSize;

        // Quantize query once per request: float → short with scale Q=4096.
        // STRIDE=16: 14 real dims + 2 zero-padded → fills Vector256<short> exactly.
        Span<short> qShort = stackalloc short[STRIDE]; // zero-initialized
        for (int d = 0; d < Dims; d++) qShort[d] = (short)(query[d] * Q);

        bool useSimd = Avx2.IsSupported && Ssse3.IsSupported;
        var qVec = Vector256.LoadUnsafe(ref qShort[0]); // load once, reuse across all clusters

        for (int p = 0; p < NPROBE; p++)
        {
            int c = centId[p];
            if (c < 0) continue;

            int start = clusterStart[c];
            int end = start + clusterSize[c];

            for (int s = start; s < end; s++)
            {
                int offset = s * STRIDE;
                int dist;

                if (useSimd)
                {
                    // Load 16 int16 reference values (14 real + 2 zero-padded).
                    var rVec = Vector256.LoadUnsafe(ref vectors[offset]);
                    // int16 subtraction: |diff| ≤ 2Q = 8192 ≤ 32767 → no overflow.
                    var diff = Avx2.Subtract(qVec, rVec);
                    // VPMADDWD: adjacent int16 pairs → int32 sum-of-squares.
                    // Each pair ≤ 2×8192² = 134M; 8 pairs ≤ 1.07B ≤ INT32_MAX.
                    var sq = Avx2.MultiplyAddAdjacent(diff, diff);
                    // Horizontal sum of 8 int32 → scalar.
                    var lo4 = sq.GetLower();
                    var sum4 = Sse2.Add(lo4, sq.GetUpper());
                    var hadd = Ssse3.HorizontalAdd(sum4, sum4);
                    dist = hadd[0] + hadd[1];
                }
                else
                {
                    dist = 0;
                    for (int d = 0; d < Dims; d++)
                    {
                        int diff = qShort[d] - vectors[offset + d];
                        dist += diff * diff;
                    }
                }

                // Find the largest distance in our top-K window — evict if current is closer.
                int maxIdx = 0;
                for (int j = 1; j < K; j++)
                    if (topDist[j] > topDist[maxIdx]) maxIdx = j;

                if (dist < topDist[maxIdx])
                {
                    topDist[maxIdx] = dist;
                    topLabel[maxIdx] = labels[s];
                }
            }
        }

        int fraudCount = 0;
        for (int j = 0; j < K; j++)
            if (topLabel[j]) fraudCount++;

        float fraudScore = fraudCount / (float)K;
        return (fraudScore, fraudScore < 0.6f);
    }
}
