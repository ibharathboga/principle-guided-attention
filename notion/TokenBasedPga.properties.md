## 1. The Derivation Process

We start with the stack of essence vectors collected from the buffer.

### Step A: The Input ()

Suppose we retrieve essence vectors, each of dimension .

- **Shape:**

### Step B: Singular Value Decomposition (SVD)

We decompose this matrix to find the underlying "Principles" (the directions of maximum variance).

- \*\*\*\*: Left singular vectors (relationships between the observations).
- \*\*\*\*: Rectangular diagonal matrix of singular values (the "strength" of each principle).
- \***\*: Right singular vectors. These are the **Principal Axes\*\* of your embedding space for this specific context.

### Step C: Truncation (The "Filter" Selection)

We don't want the whole space; we want the _Essence_. We select the top columns of that correspond to the largest singular values.

- **Let ** be these basis vectors.

### Step D: Constructing

The Principle Matrix is the orthogonal projection operator onto the subspace spanned by .

---

## 2. The Shape and Nature of

- **The Shape:**
- Even though it is derived from a smaller stack () and a reduced rank (), the final matrix must match the dimensionality of your embedding space () so it can be applied to the vectors.

- **The Properties:**
- **Symmetric:** .
- **Idempotent:** (Applying the "truth filter" twice doesn't change the result; once it's filtered, it stays filtered).
- **Rank-Deficient:** Its rank is . It has a large "null space"—any signal pointing in those directions is multiplied by zero.

---

## 3. How It Is Used (The "Live" Application)

In a standard transformer, are calculated as:

In PGA, we "wrap" the weights or the resulting vectors with the Principle Matrix:

Because is , the output maintains the original model dimensions, but the **content** is strictly confined to the -dimensional subspace you distilled from .

---

## Summary Table: Shapes

| Component             | Variable | Shape | Logic                                      |
| --------------------- | -------- | ----- | ------------------------------------------ |
| **Observation Stack** |          |       | observations of size .                     |
| **Basis Vectors**     |          |       | The "True Directions."                     |
| **Principle Matrix**  |          |       | The operator that filters the signal.      |
| **Filtered Query**    |          |       | tokens projected into the "True" subspace. |
