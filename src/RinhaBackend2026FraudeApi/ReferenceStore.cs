using System.IO.Compression;
using System.Text.Json;

internal static class ReferenceStore
{
    // Flat row-major layout: vector i starts at i*14
    // System.Half (2 bytes) instead of float (4 bytes) → 84 MB vs 168 MB per instance
    // Sentinel -1.0 is exactly representable in Half
    internal static Half[] Vectors = [];
    internal static bool[] Labels = [];
    internal static int Count;

    internal static async Task LoadAsync(string resourcesPath)
    {
        const int MaxVectors = 3_000_000;
        const int Dims = 14;

        var vectors = new Half[MaxVectors * Dims];
        var labels = new bool[MaxVectors];

        var path = Path.Combine(resourcesPath, "references.json.gz");
        await using var fileStream = File.OpenRead(path);
        await using var gzip = new GZipStream(fileStream, CompressionMode.Decompress);

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        int count = 0;

        await foreach (var entry in JsonSerializer.DeserializeAsyncEnumerable<ReferenceEntry>(gzip, options))
        {
            if (entry is null) continue;

            int offset = count * Dims;
            var v = entry.Vector;
            for (int d = 0; d < Dims; d++)
                vectors[offset + d] = (Half)v[d];

            labels[count] = entry.Label == "fraud";
            count++;
        }

        Vectors = vectors;
        Labels = labels;
        Count = count;

        Console.WriteLine($"[ReferenceStore] Loaded {count:N0} vectors");
    }

    private sealed class ReferenceEntry
    {
        public float[] Vector { get; set; } = [];
        public string Label { get; set; } = "";
    }
}
