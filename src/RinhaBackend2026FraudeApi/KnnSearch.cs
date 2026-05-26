internal static class KnnSearch
{
    private const int K = 5;
    private const int Dims = 14;

    // Squared Euclidean distance — no sqrt needed; relative order is preserved.
    // Top-K maintained as a max-heap in a fixed-size stack array:
    //   root = largest distance in current candidates → cheapest to evict.
    // stackalloc: zero heap allocation per request, no GC pressure.
    internal static (float fraudScore, bool approved) Search(float[] query)
    {
        Span<float> topDist = stackalloc float[K];
        Span<bool> topLabel = stackalloc bool[K];

        topDist.Fill(float.MaxValue);

        var vectors = ReferenceStore.Vectors;
        var labels = ReferenceStore.Labels;
        int count = ReferenceStore.Count;

        for (int i = 0; i < count; i++)
        {
            int offset = i * Dims;

            float dist = 0f;
            for (int d = 0; d < Dims; d++)
            {
                float diff = query[d] - (float)vectors[offset + d];
                dist += diff * diff;
            }

            // Find the slot with the largest distance in our top-K window.
            // For K=5, linear scan (4 comparisons) beats heap complexity.
            int maxIdx = 0;
            for (int j = 1; j < K; j++)
                if (topDist[j] > topDist[maxIdx]) maxIdx = j;

            if (dist < topDist[maxIdx])
            {
                topDist[maxIdx] = dist;
                topLabel[maxIdx] = labels[i];
            }
        }

        int fraudCount = 0;
        for (int j = 0; j < K; j++)
            if (topLabel[j]) fraudCount++;

        float fraudScore = fraudCount / (float)K;
        return (fraudScore, fraudScore < 0.6f);
    }
}
