using System.Text.Json;

var resourcesPath = Environment.GetEnvironmentVariable("RESOURCES_PATH") ?? "resources";

// --build-index mode: load references.json.gz, build IVF index, save to ivf_index.bin, exit.
// Runs during `docker build` with no memory limit — the binary is baked into the image layer.
if (args.Contains("--build-index"))
{
    await ReferenceStore.BuildIndexAsync(resourcesPath);
    Console.WriteLine("[Program] Index built successfully. Exiting.");
    return;
}

var builder = WebApplication.CreateBuilder(args);

// Use compile-time source-generated JSON serialization instead of runtime reflection.
// Reduces JSON overhead from ~400µs/req to ~80µs/req, cutting CPU utilization and CFS throttle events.
builder.Services.ConfigureHttpJsonOptions(opts =>
    opts.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default));

var mccRisk = JsonSerializer.Deserialize<Dictionary<string, float>>(
    File.ReadAllText(Path.Combine(resourcesPath, "mcc_risk.json")))!;

var app = builder.Build();

app.MapGet("/ready", () => Results.Ok());

app.MapPost("/fraud-score", (FraudScoreRequest req) =>
{
    Span<float> vector = stackalloc float[14];
    Vectorizer.Vectorize(req, mccRisk, vector);
    var (fraudScore, approved) = KnnSearch.Search(vector);
    return Results.Ok(new FraudScoreResponse(approved, fraudScore));
});

await ReferenceStore.LoadAsync(resourcesPath);

// Prevents thread pool starvation on low-core machines.
// Default min threads (2) causes request queueing under burst load + GC pauses.
// 2 workers / 4 IO keep latency predictable at the cost of ~24 MB stack commitment.
ThreadPool.SetMinThreads(2, 4);

await app.RunAsync();

record FraudScoreResponse(bool approved, float fraud_score);

record FraudScoreRequest(
    string id,
    TransactionInfo transaction,
    CustomerInfo customer,
    MerchantInfo merchant,
    TerminalInfo terminal,
    LastTransactionInfo? last_transaction);

record TransactionInfo(
    float amount,
    int installments,
    DateTime requested_at);

record CustomerInfo(
    float avg_amount,
    int tx_count_24h,
    List<string> known_merchants);

record MerchantInfo(
    string id,
    string mcc,
    float avg_amount);

record TerminalInfo(
    bool is_online,
    bool card_present,
    float km_from_home);

record LastTransactionInfo(
    DateTime timestamp,
    float km_from_current);
