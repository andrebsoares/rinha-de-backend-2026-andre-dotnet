internal static class KnnSearch
{
    private const int K = 5;
    private const int Dims = 14;
    // Compile-time alias — stays in sync with ReferenceStore without a runtime read.
    private const int NPROBE = ReferenceStore.NPROBE;

    internal static (float fraudScore, bool approved) Search(float[] query)
    {
        // Stage 1: find the NPROBE nearest centroids.
        // 200 centroids × 14 dims = 2800 distance ops (~2µs).
        // stackalloc: zero heap allocation, no GC pressure per request.
        Span<float> centDist = stackalloc float[NPROBE];
        Span<int> centId = stackalloc int[NPROBE];
        centDist.Fill(float.MaxValue);
        centId.Fill(-1);

        var centroids = ReferenceStore.Centroids;
        int kClusters = ReferenceStore.K_CLUSTERS;

        for (int c = 0; c < kClusters; c++)
        {
            int cOff = c * Dims;
            float dist = 0f;
            for (int d = 0; d < Dims; d++)
            {
                float diff = query[d] - centroids[cOff + d];
                dist += diff * diff;
            }

            // Linear max-scan to find the eviction slot (NPROBE=10 → 9 compares; beats heap).
            int maxIdx = 0;
            for (int j = 1; j < NPROBE; j++)
                if (centDist[j] > centDist[maxIdx]) maxIdx = j;

            if (dist < centDist[maxIdx])
            {
                centDist[maxIdx] = dist;
                centId[maxIdx] = c;
            }
        }

        // Stage 2: scan vectors in the NPROBE selected clusters.
        // ~NPROBE × avg_cluster_size = 10 × 15K = 150K vectors vs 3M brute-force (~20× speedup).
        Span<float> topDist = stackalloc float[K];
        Span<bool> topLabel = stackalloc bool[K];
        topDist.Fill(float.MaxValue);

        var vectors = ReferenceStore.Vectors;
        var labels = ReferenceStore.Labels;
        var invertedIndex = ReferenceStore.InvertedIndex;
        var clusterStart = ReferenceStore.ClusterStart;
        var clusterSize = ReferenceStore.ClusterSize;

        for (int p = 0; p < NPROBE; p++)
        {
            int c = centId[p];
            if (c < 0) continue;

            int start = clusterStart[c];
            int end = start + clusterSize[c];

            for (int s = start; s < end; s++)
            {
                int i = invertedIndex[s];
                int offset = i * Dims;

                float dist = 0f;
                for (int d = 0; d < Dims; d++)
                {
                    float diff = query[d] - (float)vectors[offset + d];
                    dist += diff * diff;
                }

                // Find the largest distance in our top-K window — evict if current is closer.
                int maxIdx = 0;
                for (int j = 1; j < K; j++)
                    if (topDist[j] > topDist[maxIdx]) maxIdx = j;

                if (dist < topDist[maxIdx])
                {
                    topDist[maxIdx] = dist;
                    topLabel[maxIdx] = labels[i];
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
