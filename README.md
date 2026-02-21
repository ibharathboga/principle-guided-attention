# Principle-Guided Attention

> **Heads Up:** I am not an expert. This repository is primarily a place for me to store and not forget a core idea I had. I wanted to entertain the idea of "mathematacising" this concept, and I used LLMs to help run the experiments. I'm not entirely sure why I made it public—maybe I'll work on it more later, maybe not. There's a slight chance someone might stumble upon this and want to look into it or build upon it, so I wanted to provide this context upfront.

## The Core Idea

At a philosophical level:
- We are always looking at a smaller piece of a larger puzzle.
- **Thoughts** are the computed results of observations intended to find principles.
- **Principles** are computed from pure observations.
- Principles are always partial because we can't observe everything.

### Technical Process

This philosophical concept translates into a modified attention mechanism:

1. **Observation to Essence:** Observations are turned into "essence vectors" through the transformer attention process.
2. **Retrieval:** We find observation essence vectors stored in a vector database that are similar or near the current query observation.
3. **Principle Extraction via SVD:** The retrieved observation essence tensor goes through Singular Value Decomposition (SVD) to extract "principles".
4. **Enhanced Attention:** This principle tensor is then fed back into the QKV (Query, Key, Value) process to enhance the questioning and answering that takes place in standard attention cycles.

---
*Feel free to explore the code, but keep in mind this is an experimental playground for a theoretical idea!*
