from pymilvus import connections, utility

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection_name = "document_collection"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✅ 已删除 collection: {collection_name}")
else:
    print("⚠️ collection 不存在")
