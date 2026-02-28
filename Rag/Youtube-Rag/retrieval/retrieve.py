# def retrieve_docs(vector_store, query, filter_condition=None, k=5):
#     if filter_condition:
#         return vector_store.similarity_search(
#             query=query,
#             k=k,
#             filter=filter_condition
#         )
#     return vector_store.similarity_search(query=query, k=k)