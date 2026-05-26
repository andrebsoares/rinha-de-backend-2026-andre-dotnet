using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

var resourcesPath = Environment.GetEnvironmentVariable("RESOURCES_PATH") ?? "resources";

var mccRisk = JsonSerializer.Deserialize<Dictionary<string, float>>(
    File.ReadAllText(Path.Combine(resourcesPath, "mcc_risk.json")))!;

var app = builder.Build();

app.MapGet("/ready", () => Results.Ok());

app.MapPost("/fraud-score", (FraudScoreRequest req) =>
{
    var vector = Vectorizer.Vectorize(req, mccRisk);
    var (fraudScore, approved) = KnnSearch.Search(vector);
    return Results.Ok(new FraudScoreResponse(approved, fraudScore));
});

await ReferenceStore.LoadAsync(resourcesPath);

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
