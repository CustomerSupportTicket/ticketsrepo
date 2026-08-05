from rag_pipeline import get_rag_response

result = get_rag_response(
    "My web portal account is disabled."
)

print("\nAnswer:\n")
print(result["answer"])

print("\nSources:\n")

for i, src in enumerate(result["sources"], 1):
    print(f"\nSource {i}")
    print(src)