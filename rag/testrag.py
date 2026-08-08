from rag_pipeline import get_rag_response

query = "Why was my Web Portal account deactivated? I need it restored urgently."

result = get_rag_response(query)

print("=" * 60)
print("ANSWER")
print("=" * 60)
print(result["answer"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, source in enumerate(result["sources"], 1):
    print(f"\nSource {i}")
    print(source)