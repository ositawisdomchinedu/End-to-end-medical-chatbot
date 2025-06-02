# Prompt template for RAG
prompt_template = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer"
    "the question. If you don't know the answer, say that you" 
    "don't know. Use three sentences maximum and keep the answer concise"
    "\n\n" 
    "Context:\n{context}"
    "\n\n" 
    "Question:\n{question}"
)

prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
)
