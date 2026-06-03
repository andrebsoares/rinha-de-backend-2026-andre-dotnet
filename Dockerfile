FROM mcr.microsoft.com/dotnet/sdk:10.0-alpine AS build
WORKDIR /src
COPY src/RinhaBackend2026FraudeApi/RinhaBackend2026FraudeApi.csproj .
RUN dotnet restore
COPY src/RinhaBackend2026FraudeApi/ .
RUN dotnet publish -c Release -o /app/publish

# Build the IVF index offline so runtime startup is a fast binary load (~2s).
# This stage runs once; Docker caches the layer — no rebuild unless source or resources change.
# No memory limit here, so the K-Means peak (~145 MB) is safe.
FROM mcr.microsoft.com/dotnet/aspnet:10.0-alpine AS index-builder
WORKDIR /app
COPY --from=build /app/publish .
COPY resources/ ./resources/
ENV RESOURCES_PATH=/app/resources
RUN dotnet RinhaBackend2026FraudeApi.dll --build-index

FROM mcr.microsoft.com/dotnet/aspnet:10.0-alpine
RUN apk add --no-cache wget
WORKDIR /app
COPY --from=build /app/publish .
COPY resources/ ./resources/
# Copy the pre-built index (baked from index-builder stage)
COPY --from=index-builder /app/resources/ivf_index.bin ./resources/ivf_index.bin
ENV RESOURCES_PATH=/app/resources
ENTRYPOINT ["dotnet", "RinhaBackend2026FraudeApi.dll"]
