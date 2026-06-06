using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// Minimal profiler para identificar gargalos
// Rode: dotnet run --project test-profiling.csproj

public class Profiler
{
    private const int N = 10000; // Número de queries de teste
    private static float[] randomVectors;
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float EuclideanDistanceSIMD(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            fixed (float* pA = a, pB = b)
            {
                var aVec1 = System.Runtime.Intrinsics.Vector256.Load(pA);
                var bVec1 = System.Runtime.Intrinsics.Vector256.Load(pB);
                var diff1 = System.Runtime.Intrinsics.X86.Avx.Subtract(aVec1, bVec1);
                var sq1 = System.Runtime.Intrinsics.X86.Avx.Multiply(diff1, diff1);
                
                var aVec2 = System.Runtime.Intrinsics.Vector256.Load(pA + 8);
                var bVec2 = System.Runtime.Intrinsics.Vector256.Load(pB + 8);
                var diff2 = System.Runtime.Intrinsics.X86.Avx.Subtract(aVec2, bVec2);
                var sq2 = System.Runtime.Intrinsics.X86.Avx.Multiply(diff2, diff2);
                
                var sum8 = System.Runtime.Intrinsics.X86.Avx.Add(sq1, sq2);
                var lo4 = sum8.GetLower();
                var hi4 = sum8.GetUpper();
                var sum4 = System.Runtime.Intrinsics.X86.Sse.Add(lo4, hi4);
                var ha1 = System.Runtime.Intrinsics.X86.Sse3.HorizontalAdd(sum4, sum4);
                var ha2 = System.Runtime.Intrinsics.X86.Sse3.HorizontalAdd(ha1, ha1);
                return ha2[0];
            }
        }
        
        // Fallback scalar
        float sum = 0;
        for (int i = 0; i < 14; i++)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int EuclideanDistanceQuantizedSIMD(ReadOnlySpan<short> a, ReadOnlySpan<short> b)
    {
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Ssse3.IsSupported)
        {
            fixed (short* pA = a, pB = b)
            {
                var aVec = System.Runtime.Intrinsics.Vector256.Load(pA);
                var bVec = System.Runtime.Intrinsics.Vector256.Load(pB);
                var diff = System.Runtime.Intrinsics.X86.Avx2.Subtract(aVec, bVec);
                var sq = System.Runtime.Intrinsics.X86.Avx2.MultiplyAddAdjacent(diff, diff);
                var lo4 = sq.GetLower();
                var sum4 = System.Runtime.Intrinsics.X86.Sse2.Add(lo4, sq.GetUpper());
                var hadd = System.Runtime.Intrinsics.X86.Ssse3.HorizontalAdd(sum4, sum4);
                return hadd[0] + hadd[1];
            }
        }
        
        // Fallback scalar
        int sum = 0;
        for (int i = 0; i < 14; i++)
        {
            int diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    public static void Main()
    {
        Console.WriteLine("=== Análise de Gargalos IVF .NET ===");
        
        // Teste 1: Throughput SIMD float
        Console.WriteLine("\n1. Throughput SIMD float (14 dims):");
        var a = new float[16]; // 14 + 2 padding
        var b = new float[16];
        var rng = new Random(42);
        for (int i = 0; i < 16; i++)
        {
            a[i] = (float)rng.NextDouble();
            b[i] = (float)rng.NextDouble();
        }
        
        long totalOps = 0;
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 1000)
        {
            for (int i = 0; i < 1000; i++)
            {
                EuclideanDistanceSIMD(a, b);
            }
            totalOps += 1000;
        }
        sw.Stop();
        Console.WriteLine($"  {totalOps:N0} ops/sec = {totalOps / 1_000_000f:F2} M ops/sec");
        Console.WriteLine($"  {1_000_000f / (totalOps / sw.Elapsed.TotalSeconds):F2} ns por distância");
        
        // Teste 2: Throughput SIMD short
        Console.WriteLine("\n2. Throughput SIMD short (14 dims quantizados):");
        var aShort = new short[16];
        var bShort = new short[16];
        for (int i = 0; i < 16; i++)
        {
            aShort[i] = (short)rng.Next(-4096, 4096);
            bShort[i] = (short)rng.Next(-4096, 4096);
        }
        
        totalOps = 0;
        sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 1000)
        {
            for (int i = 0; i < 1000; i++)
            {
                EuclideanDistanceQuantizedSIMD(aShort, bShort);
            }
            totalOps += 1000;
        }
        sw.Stop();
        Console.WriteLine($"  {totalOps:N0} ops/sec = {totalOps / 1_000_000f:F2} M ops/sec");
        Console.WriteLine($"  {1_000_000f / (totalOps / sw.Elapsed.TotalSeconds):F2} ns por distância");
        
        // Teste 3: Cache throughput
        Console.WriteLine("\n3. Estimativa de throughput com cache misses:");
        long memorySize = 100_000 * 16 * 2; // 100k vetores × 16 dims × 2 bytes
        long l3CacheSize = 8_000_000; // 8 MB típico
        Console.WriteLine($"  Dataset: {memorySize / 1_000_000} MB");
        Console.WriteLine($"  L3 cache típico: {l3CacheSize / 1_000_000} MB");
        Console.WriteLine($"  Cache miss rate estimado: {Math.Max(0, 100f * (memorySize - l3CacheSize) / memorySize):F1}%");
        
        // Teste 4: GC pressure
        Console.WriteLine("\n4. Pressão no GC:");
        Console.WriteLine($"  Gen0: {GC.CollectionCount(0)}");
        Console.WriteLine($"  Gen1: {GC.CollectionCount(1)}");
        Console.WriteLine($"  Gen2: {GC.CollectionCount(2)}");
        
        // Teste 5: SIMD suporte
        Console.WriteLine("\n5. Suporte SIMD:");
        Console.WriteLine($"  Avx2: {System.Runtime.Intrinsics.X86.Avx2.IsSupported}");
        Console.WriteLine($"  Sse3: {System.Runtime.Intrinsics.X86.Sse3.IsSupported}");
        Console.WriteLine($"  Ssse3: {System.Runtime.Intrinsics.X86.Ssse3.IsSupported}");
        
        Console.WriteLine("\n=== Conclusões ===");
        Console.WriteLine("Use 'dotnet publish -c Release -r linux-x64 --self-contained' para AOT");
        Console.WriteLine("Considere Native AOT para reduzir JIT overhead em primeira execução");
    }
}