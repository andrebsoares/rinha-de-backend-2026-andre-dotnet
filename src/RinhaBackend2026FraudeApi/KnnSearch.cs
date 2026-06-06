using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

internal static class KnnSearch
{
    private const int K = 5;
    private const int Dims = 14;
    private const int STRIDE = ReferenceStore.STRIDE;
    private const int Q = ReferenceStore.Q;
    private const int NPROBE = ReferenceStore.NPROBE;

    internal static (float fraudScore, bool approved) Search(ReadOnlySpan<float> query)
    {
        // Stage 1: find NPROBE nearest centroids (NO SORTING - just max-scan)
        Span<float> centDist = stackalloc float[NPROBE];
        Span<int> centId = stackalloc int[NPROBE];
        centDist.Fill(float.MaxValue);
        centId.Fill(-1);

        var centroids = ReferenceStore.Centroids;
        int kClusters = ReferenceStore.K_CLUSTERS;
        int strideFlt = ReferenceStore.STRIDE_FLOAT;

        if (Avx2.IsSupported)
        {
            Span<float> qPad = stackalloc float[ReferenceStore.STRIDE_FLOAT];
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

                // Max-scan eviction (NO SORTING)
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

        // Stage 2: scan vectors in NPROBE clusters (NO SORTING, NO PRUNING, NO EARLY REJECTION)
        Span<int> topDist = stackalloc int[K];
        Span<bool> topLabel = stackalloc bool[K];
        topDist.Fill(int.MaxValue);

        var vectors = ReferenceStore.Vectors;
        var labels = ReferenceStore.Labels;
        var clusterStart = ReferenceStore.ClusterStart;
        var clusterSize = ReferenceStore.ClusterSize;

        ref short vectorsRef = ref MemoryMarshal.GetArrayDataReference(vectors);
        ref bool labelsRef = ref MemoryMarshal.GetArrayDataReference(labels);
        ref int clusterStartRef = ref MemoryMarshal.GetArrayDataReference(clusterStart);
        ref int clusterSizeRef = ref MemoryMarshal.GetArrayDataReference(clusterSize);

        // Quantize query
        Span<short> qShort = stackalloc short[STRIDE];
        for (int d = 0; d < Dims; d++) qShort[d] = (short)(query[d] * Q);

        bool useSimd = Avx2.IsSupported && Ssse3.IsSupported;
        var qVec = Vector256.LoadUnsafe(ref qShort[0]);

        for (int p = 0; p < NPROBE; p++)
        {
            int c = centId[p];
            if (c < 0) continue;

            int start = Unsafe.Add(ref clusterStartRef, c);
            int size = Unsafe.Add(ref clusterSizeRef, c);
            int end = start + size;

            for (int s = start; s < end; s++)
            {
                int offset = s * STRIDE;
                int dist;

                if (useSimd)
                {
                    ref short vecRef = ref Unsafe.Add(ref vectorsRef, offset);
                    var rVec = Vector256.LoadUnsafe(ref vecRef);
                    var diff = Avx2.Subtract(qVec, rVec);
                    var sq = Avx2.MultiplyAddAdjacent(diff, diff);
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
                        int diff = qShort[d] - Unsafe.Add(ref vectorsRef, offset + d);
                        dist += diff * diff;
                    }
                }

                // Simple max-scan for worst distance (NO UNROLLING, NO EARLY REJECTION)
                int maxIdx = 0;
                int maxDist = topDist[0];
                for (int j = 1; j < K; j++)
                    if (topDist[j] > maxDist) { maxIdx = j; maxDist = topDist[j]; }

                if (dist < maxDist)
                {
                    topDist[maxIdx] = dist;
                    topLabel[maxIdx] = Unsafe.Add(ref labelsRef, s);
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
