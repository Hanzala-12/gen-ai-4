We built four generative models from scratch and put them in one web app:

- A masked autoencoder on satellite images: with only 160 labels it reaches 46.6% accuracy (chance is 10%)
- A VAE that samples and morphs handwritten Urdu characters
- A diffusion model that writes a requested Urdu character from pure noise
- A RAG system that answers biomedical questions with cited PubMed passages

The biggest lesson: retrieval quality limits everything in RAG. Our retriever missed on 14% of questions, and the generator sometimes answered confidently anyway.

Full write-up: BLOG_URL
Code and results: https://github.com/Hanzala-12/gen-ai-4

#GenerativeAI #DeepLearning #Diffusion #RAG #PyTorch
