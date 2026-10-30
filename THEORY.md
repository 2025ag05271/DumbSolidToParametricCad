# Theoretical Background: Learning Parametric CAD Histories from B-Rep Solids

This document lists the theory a developer should know to understand a system that takes a
boundary-representation (B-Rep) solid with no feature history and predicts a parametric feature
history that rebuilds it. Topics are named and briefly explained, not taught. For each topic, the
**search terms** are good starting points for textbooks, lecture notes and papers. Each section
ends with a **study prompt** you can paste into an LLM for a quick guided introduction.

The problem sits where three fields meet:

1. **Computational geometry and solid modelling.** How a CAD kernel represents and builds shapes.
2. **Geometric deep learning.** How neural networks process non-Euclidean data such as graphs and surfaces.
3. **Sequence modelling.** How Transformers generate structured, program-like output.

---

## How to use this guide

- Work through the sections in order. Later sections assume the earlier ones.
- For a given topic, search the listed terms and read one textbook chapter or lecture. Then use
  the section prompt to check your understanding.
- **Generic study prompt** (works for any topic in this document):

  > I am an engineer learning *[TOPIC]* to understand a deep-learning system that converts CAD
  > B-Rep solids into parametric feature histories. Explain *[TOPIC]* in about 15 minutes of
  > reading: give the intuition first, then the key definitions and equations, a small worked
  > example, and the common pitfalls. Finish with 5 self-test questions with answers, and tell me
  > which prerequisite topics I should know first.

---

## Part A — Mathematical foundations

| Topic | Why it matters | Search terms |
|---|---|---|
| Vectors, matrices, linear maps | All network layers and all geometric transforms are linear algebra | "linear algebra for machine learning", "matrix multiplication intuition" |
| Coordinate frames and rigid transforms | Sketch planes, local and world coordinates, mirroring | "rigid body transformation", "homogeneous coordinates", "change of basis" |
| Dot and cross products, orthogonality | Normals, plane frames, orientation tests | "cross product geometric meaning", "orthonormal basis" |
| Multivariable calculus, partial derivatives | Surface tangents and normals; gradients for training | "partial derivatives", "Jacobian matrix", "chain rule multivariable" |
| Probability, categorical distribution | The model outputs a probability over the next token | "categorical distribution", "softmax function", "maximum likelihood estimation" |
| Information theory basics | The training loss | "cross-entropy loss", "KL divergence", "perplexity" |
| Numerical precision | Tolerances in kernels and low-precision training | "floating point arithmetic", "machine epsilon", "bfloat16 vs float16" |
| Monte Carlo integration | Estimating volumes and IoU by random sampling | "Monte Carlo integration", "Monte Carlo volume estimation error" |

**Study prompt:**
> Teach me the minimum linear algebra, multivariable calculus and probability needed to follow
> (a) how neural networks are trained with gradient descent, and (b) how CAD curves and surfaces
> are defined parametrically. Use small numeric examples, and cover coordinate frames, the cross
> product, the Jacobian, the softmax function and cross-entropy.

---

## Part B — Neural network fundamentals

| Topic | Search terms |
|---|---|
| Perceptron and multilayer perceptron (MLP) | "multilayer perceptron", "universal approximation theorem" |
| Activation functions | "ReLU vs GELU vs LeakyReLU" |
| Loss functions, empirical risk minimisation | "empirical risk minimization", "cross-entropy classification loss" |
| Backpropagation and automatic differentiation | "backpropagation derivation", "reverse-mode automatic differentiation" |
| Optimisers | "stochastic gradient descent", "Adam optimizer", "AdamW decoupled weight decay" |
| Learning-rate schedules | "learning rate warmup", "cosine annealing schedule" |
| Initialisation | "Xavier initialization", "GPT-2 weight initialization std 0.02" |
| Normalisation layers | "batch normalization", "layer normalization", "pre-LN vs post-LN transformer" |
| Residual connections | "ResNet residual connection", "vanishing gradient problem" |
| Regularisation | "dropout", "weight decay", "label smoothing", "gradient clipping" |
| Generalisation and overfitting | "bias-variance tradeoff", "overfitting validation loss", "train/val/test split" |
| Mixed-precision training | "automatic mixed precision training", "bfloat16 training" |

**Study prompt:**
> Explain how a neural network is trained end to end: a forward pass through an MLP, a
> cross-entropy loss, backpropagation, and AdamW updates with warm-up and cosine learning-rate
> decay. Then explain LayerNorm, residual connections, dropout and weight decay, and why each one
> helps deep networks train stably. Explain how to tell overfitting from a training curve, and
> why validation loss can rise while the quality of generated outputs still improves.

---

## Part C — Convolutional networks on structured grids

| Topic | Search terms |
|---|---|
| 1D and 2D convolution, kernels, channels | "convolution neural network explained", "1D convolution for sequences" |
| Padding, stride, receptive field | "receptive field CNN", "same padding" |
| Pooling and global average pooling | "global average pooling" |
| Weight sharing and translation equivariance | "translation equivariance CNN" |
| CNNs on non-image grids | "CNN on parameter-space grids", "UV-Net surface CNN" |

**Study prompt:**
> Explain 1D and 2D convolutional neural networks, covering kernels, channels, padding, receptive
> field and global pooling. Then explain how a CNN can process a surface sampled on a regular
> (u, v) parameter grid, where each grid cell stores a 3D point, a normal and an inside/outside
> mask, instead of an image. What invariances do we get, and which do we lose?

---

## Part D — Graph neural networks (GNNs)

| Topic | Search terms |
|---|---|
| Graph basics: nodes, edges, adjacency, degree | "graph theory basics adjacency matrix" |
| Directed versus undirected graphs | "directed graph message passing" |
| Message passing neural networks (MPNN) | "message passing neural network Gilmer" |
| Aggregation functions | "GNN aggregation sum mean max", "permutation invariance" |
| GCN, GraphSAGE, GAT, GIN | "graph convolutional network Kipf", "graph attention network", "graph isomorphism network" |
| Edge features and edge updates | "edge-conditioned convolution", "GNN with edge features", "interaction networks" |
| Batching many graphs | "mini-batching graphs disjoint union" |
| Over-smoothing and expressivity limits | "GNN oversmoothing", "Weisfeiler-Lehman test GNN" |
| Graph Transformers | "graph transformer", "transformer as fully connected GNN" |
| Geometric deep learning (the unifying view) | "geometric deep learning Bronstein grids groups graphs geodesics gauges" |

**Study prompt:**
> Teach me graph neural networks from first principles: how message passing works, why
> aggregation must be permutation-invariant, how edge features enter the messages, and how a
> batch of graphs is processed as one disjoint graph. Compare GCN, GAT and GIN, then explain
> over-smoothing and the Weisfeiler-Lehman limit. Finally, explain why a Transformer over graph
> nodes is like a GNN on a fully connected graph.

---

## Part E — Sequence modelling and Transformers

| Topic | Search terms |
|---|---|
| Tokenisation and vocabularies | "tokenization in language models", "discrete tokens for continuous values" |
| Embeddings and weight tying | "token embedding", "tied input output embeddings" |
| Autoregressive modelling | "autoregressive language model next token prediction" |
| Attention mechanism | "scaled dot-product attention", "query key value intuition" |
| Multi-head attention | "multi-head attention" |
| Positional encoding | "sinusoidal positional encoding", "learned positional embeddings", "RoPE" |
| Causal masking | "causal mask transformer decoder" |
| Transformer encoder and decoder blocks | "Attention Is All You Need", "transformer architecture explained" |
| Self-attention versus cross-attention | "cross attention encoder decoder" |
| Teacher forcing and exposure bias | "teacher forcing", "exposure bias sequence generation" |
| KV caching for fast inference | "KV cache transformer inference" |
| Attention complexity | "quadratic attention complexity", "key padding mask" |

**Study prompt:**
> Explain the Transformer architecture from "Attention Is All You Need" step by step: token and
> positional embeddings, scaled dot-product attention (with the equation), multi-head attention,
> causal masking, pre-norm residual blocks, and the difference between self-attention and
> cross-attention. Then explain autoregressive training with teacher forcing, exposure bias, and
> how a KV cache speeds up generation. Work a tiny attention example with 3 tokens by hand.

---

## Part F — Encoder–decoder (sequence-to-sequence) architectures

| Topic | Search terms |
|---|---|
| Seq2seq and conditional generation | "sequence to sequence learning", "conditional language model" |
| Encoder memory and cross-attention conditioning | "encoder decoder transformer memory" |
| Set or graph encoders feeding a sequence decoder | "graph-to-sequence", "set transformer", "Perceiver" |
| Global or summary tokens | "CLS token", "global token transformer" |
| Multimodal analogues | "image captioning transformer", "speech-to-text encoder decoder" |

**Study prompt:**
> Explain encoder–decoder (seq2seq) Transformer architectures. How does the decoder use
> cross-attention to read the encoder's output memory? How does this design change when the input
> is not a sequence but a set or graph of variable size, as in image captioning, graph-to-sequence
> models or the Perceiver? What is the role of a global "summary" token, and how are padding masks
> used when examples in a batch have different numbers of input elements?

---

## Part G — Decoding, structured output and evaluation

| Topic | Search terms |
|---|---|
| Greedy decoding, sampling, temperature | "greedy decoding vs sampling", "softmax temperature" |
| Beam search and top-k / nucleus sampling | "beam search", "nucleus sampling top-p" |
| Grammar-constrained decoding | "grammar constrained decoding", "logit masking", "constrained generation context-free grammar" |
| Quantisation of continuous values into tokens | "quantization of coordinates as tokens", "discretization regression as classification" |
| Canonicalisation and output ambiguity | "canonical form program synthesis", "multiple valid outputs one-to-many mapping" |
| Program synthesis and inverse procedural modelling | "neural program synthesis", "inverse procedural modeling", "inverse CSG" |
| Generate-and-verify / reranking | "sample and rerank", "execution-guided decoding", "self-consistency" |
| Reinforcement learning fine-tuning with rewards | "policy gradient REINFORCE", "PPO", "GRPO", "RL from execution feedback" |
| Shape similarity metrics | "intersection over union 3D", "Chamfer distance", "Hausdorff distance" |
| Synthetic data and domain gap | "procedural data generation", "sim-to-real domain gap", "domain randomization" |

**Study prompt:**
> I have a Transformer that generates a token sequence describing a CAD construction program.
> Explain greedy decoding, temperature sampling, beam search and nucleus sampling. Explain how
> grammar-constrained decoding masks invalid tokens at each step. Explain why continuous numbers
> are quantised into discrete tokens and what precision trade-offs that creates. Explain why
> generating several candidates and keeping the one whose executed result best matches the input
> (execution-guided reranking) helps, and how reinforcement learning with a geometric reward
> (e.g. IoU) such as GRPO could fine-tune the model further. Define IoU and Chamfer distance.

---

## Part H — Solid modelling representations

| Topic | Why it matters | Search terms |
|---|---|---|
| Solid modelling paradigms | Choosing a representation | "B-Rep vs CSG vs mesh vs voxel", "solid modeling Requicha" |
| Boundary representation (B-Rep) | The input format | "boundary representation solid modeling" |
| Topology versus geometry | The B-Rep separates connectivity from shape | "B-Rep topology vs geometry" |
| Topological entities | Solid, shell, face, loop/wire, coedge, edge, vertex | "B-Rep topological entities", "half-edge data structure", "winged-edge data structure", "coedge" |
| Manifold and non-manifold solids | Validity of a solid | "2-manifold solid", "non-manifold topology" |
| Euler–Poincaré formula | Topological consistency check | "Euler-Poincaré formula V-E+F", "Euler operators" |
| Orientation and face normals | Inside versus outside | "face orientation B-Rep", "outward normal convention" |
| Face adjacency graph | How a B-Rep becomes a GNN input | "face adjacency graph CAD", "attributed adjacency graph feature recognition" |
| Constructive Solid Geometry (CSG) | Booleans as a construction language | "constructive solid geometry", "CSG tree" |
| Feature-based parametric modelling | The output format (history tree) | "feature-based modeling", "parametric history tree", "design intent" |
| Sketch and constraint solving | 2D profiles behind pads and pockets | "geometric constraint solving sketch", "degrees of freedom sketch constraints" |
| Persistent / topological naming problem | Why referencing edges across rebuilds is hard | "persistent naming problem CAD", "topological naming problem" |
| Feature recognition | Classical, non-ML relative of this task | "automatic feature recognition CAD", "feature recognition from B-Rep" |
| Exchange formats | How dumb solids are delivered | "STEP ISO 10303 AP203 AP214 AP242", "IGES", "dumb solid vs native CAD" |

**Study prompt:**
> Explain boundary representation (B-Rep) solid modelling. Cover the hierarchy solid → shell →
> face → loop → coedge → edge → vertex, the difference between topology and geometry, manifold
> validity, the Euler–Poincaré formula, and face orientation. Then compare B-Rep with CSG, meshes
> and feature-based parametric history trees. Explain what a STEP "dumb solid" contains and what
> is lost compared with a native CAD file. Finally, explain the persistent (topological) naming
> problem and why it makes referencing edges for fillets difficult.

---

## Part I — Parametric curves and surfaces

Geometry in a kernel is parametric. A curve maps one parameter to 3D space, and a surface maps two:

$$\mathbf{C}(t): [t_0, t_1] \to \mathbb{R}^3, \qquad \mathbf{S}(u, v): [u_0,u_1]\times[v_0,v_1] \to \mathbb{R}^3$$

| Topic | Key equation or idea (names to search) | Search terms |
|---|---|---|
| Parametric versus implicit representation | $\mathbf{S}(u,v)$ versus $F(x,y,z)=0$ | "parametric vs implicit surfaces" |
| Lines and planes | $\mathbf{C}(t)=\mathbf{P}_0+t\,\mathbf{d}$; $\mathbf{S}(u,v)=\mathbf{O}+u\,\mathbf{X}+v\,\mathbf{Y}$ | "parametric line plane equation" |
| Conics: circle, ellipse | $\mathbf{C}(\theta)=\mathbf{c}+r(\cos\theta\,\mathbf{X}+\sin\theta\,\mathbf{Y})$ | "parametric circle 3D", "conic sections rational representation" |
| Analytic surfaces | Plane, cylinder, cone, sphere, torus | "analytic surfaces CAD quadrics torus parametrization" |
| Swept surfaces | Extrusion $\mathbf{S}(u,v)=\mathbf{C}(u)+v\,\mathbf{d}$; revolution about an axis | "surface of extrusion", "surface of revolution", "sweep surface" |
| Bézier curves | Bernstein basis $B_{i,n}(t)=\binom{n}{i}t^i(1-t)^{n-i}$; de Casteljau algorithm | "Bezier curve", "Bernstein polynomials", "de Casteljau algorithm" |
| B-spline curves | $\mathbf{C}(t)=\sum_i N_{i,p}(t)\,\mathbf{P}_i$; Cox–de Boor recursion; knot vector | "B-spline basis functions", "Cox-de Boor recursion", "knot vector", "local support" |
| NURBS | Rational form $\mathbf{C}(t)=\dfrac{\sum_i N_{i,p}(t)\,w_i\mathbf{P}_i}{\sum_i N_{i,p}(t)\,w_i}$; exact conics | "NURBS curve", "rational B-spline weights", "NURBS circle representation" |
| Tensor-product surfaces | $\mathbf{S}(u,v)=\sum_i\sum_j N_{i,p}(u)N_{j,q}(v)\,w_{ij}\mathbf{P}_{ij}\,/\,\sum\sum N N w$ | "tensor product NURBS surface" |
| Knot insertion, degree elevation | Editing splines without changing shape | "knot insertion", "degree elevation B-spline" |
| Offset curves and surfaces | Basis of fillets, shells, tool paths | "offset surface", "offset curve self-intersection" |

**Study prompt:**
> Teach me parametric curves and surfaces as used in CAD kernels. Start with parametric lines,
> circles and planes. Then cover Bézier curves (Bernstein basis, de Casteljau), B-splines (basis
> functions, Cox–de Boor recursion, knot vectors, local support), NURBS (rational weights, exact
> circles), and tensor-product surfaces. Explain extrusion and revolution surfaces and offset
> surfaces. Give the equations, a small numeric example of evaluating a quadratic Bézier curve,
> and explain why industry uses NURBS. Recommend the standard textbook ("The NURBS Book").

---

## Part J — Differential geometry of curves and surfaces

| Topic | Search terms |
|---|---|
| Tangent vector, arc length, reparameterisation | "arc length parametrization", "regular curve" |
| Curvature and torsion; Frenet–Serret frame | "curvature of a space curve", "Frenet-Serret formulas" |
| Surface partial derivatives and normal | "surface normal cross product of partial derivatives" ($\mathbf{n}=\mathbf{S}_u\times\mathbf{S}_v/\lVert\mathbf{S}_u\times\mathbf{S}_v\rVert$) |
| First fundamental form (metric) | "first fundamental form", "metric tensor surface" |
| Second fundamental form | "second fundamental form", "shape operator" |
| Principal, Gaussian and mean curvature | "principal curvatures", "Gaussian curvature", "mean curvature" |
| Singular points and degeneracies | "degenerate surface parametrization pole of sphere", "singular point parametric surface" |

**Study prompt:**
> Give me a practical introduction to the differential geometry of curves and surfaces for CAD:
> tangent vectors, arc length, curvature and torsion (Frenet–Serret), the surface normal from
> partial derivatives, and the first and second fundamental forms. Cover principal, Gaussian and
> mean curvature, with values for a plane, cylinder, sphere and torus. Explain where
> parameterisations become singular (e.g. the pole of a sphere, the apex of a cone) and why that
> matters for sampling.

---

## Part K — Parameter space, UV mapping and trimming

| Topic | Why it matters | Search terms |
|---|---|---|
| (u, v) parameter domain of a surface | Every face is "drawn" on a 2D parameter sheet | "surface parameter space uv domain" |
| Trimmed surfaces | A face is a surface restricted by boundary loops | "trimmed NURBS surface", "trimming curves" |
| Parameter-space curves (p-curves) | Each edge has a 2D image in each adjacent face's (u, v) space | "pcurve parametric space curve on surface", "curve on surface" |
| Point inversion / projection | Finding (u, v) for a 3D point | "point inversion NURBS", "closest point projection onto surface" |
| Point classification (in / on / out) | Trim mask; point-in-solid tests | "point in polygon winding number", "point membership classification solid" |
| Periodic and closed surfaces, seams | Cylinders and spheres wrap around | "periodic surface seam edge" |
| Uniform sampling in parameter space | How UV grids are built | "uniform parametric sampling distortion" |
| Parameterisation distortion | Equal (u, v) steps ≠ equal 3D distances | "parameterization distortion", "chord length parametrization" |
| UV-Net representation | Faces as (u, v) grids of points, normals and a trim mask; edges as u-grids | "UV-Net learning from boundary representations" |
| Graphics UV mapping (for contrast) | Same name, different purpose (texture coordinates on meshes) | "UV mapping texture coordinates", "mesh parameterization" |

**Study prompt:**
> Explain the (u, v) parameter space of CAD surfaces and how it differs from texture UV mapping
> in computer graphics. Cover trimmed surfaces, trimming loops, p-curves (an edge's 2D
> representation on each adjacent face), periodic surfaces and seam edges, point inversion, and
> in/out classification of a (u, v) point. Then explain the UV-Net representation (CVPR 2021): each
> face sampled on an N×N grid of (point, normal, trim-mask) and each edge as an N-point curve grid.
> Discuss what distortion uniform parameter sampling introduces.

---

## Part L — Continuity: parametric (Cⁿ) and geometric (Gⁿ)

| Level | Meaning | Visual effect |
|---|---|---|
| **C⁰ / G⁰** | Positional: the pieces touch | Visible crease |
| **G¹** | Tangent directions match (tangent continuity) | No crease, but reflections jump |
| **C¹** | First derivatives equal (magnitude too); parameter-dependent | — |
| **G²** | Curvature matches (curvature continuity) | Smooth reflections; a typical fillet is only G¹ |
| **C²** | Second derivatives equal; cubic B-splines are C² at simple knots | — |
| **G³** | Rate of change of curvature matches | Required for Class-A automotive surfaces |

Search terms: "parametric vs geometric continuity", "G1 G2 G3 continuity CAD", "Cn continuity
B-spline knot multiplicity", "curvature comb", "zebra stripe analysis", "reflection lines",
"Class-A surfacing", "rolling-ball fillet G1 vs curvature-continuous blend".

**Study prompt:**
> Explain parametric continuity (C⁰, C¹, C², C³) and geometric continuity (G⁰, G¹, G², G³) between
> curves and between surfaces, with the mathematical conditions for each and a 2D example of two
> Bézier segments. Explain why Gⁿ is independent of parameterisation, how knot multiplicity
> controls continuity in B-splines, and why a constant-radius fillet is only G¹ to its neighbours.
> Explain how designers check continuity with curvature combs, zebra stripes and reflection lines,
> and what "Class-A" surfaces require.

---

## Part M — Inside a geometric modelling kernel

| Topic | What the kernel does | Search terms |
|---|---|---|
| Kernel architecture | Geometry library + topology + algorithms + tolerances | "geometric modeling kernel Parasolid ACIS Open CASCADE CGM" |
| Primitive and feature construction | Extrude (pad), cut-extrude (pocket), revolve, holes | "B-Rep extrusion algorithm", "sweep operation solid modeling" |
| Boolean operations | Union, subtraction, intersection of solids | "B-Rep Boolean operations algorithm", "regularized Boolean operations" |
| Surface–surface intersection (SSI) | The core of Booleans | "surface surface intersection marching method", "subdivision intersection" |
| Curve–surface and curve–curve intersection | Edge splitting, trimming | "curve surface intersection Newton iteration" |
| Fillets and blends | Rolling-ball, variable-radius, setbacks/corners | "rolling ball blend", "fillet surface generation B-Rep", "vertex blend" |
| Chamfers | Distance–distance, distance–angle | "chamfer B-Rep algorithm" |
| Shelling and offsetting | Wall thickening | "solid shelling offset B-Rep" |
| Tolerant modelling | Gaps between pieces are allowed within a tolerance | "tolerant modeling", "geometric tolerance B-Rep edges vertices" |
| Robustness | Near-coincident faces, slivers, numerical failure | "robust geometric computation", "sliver faces CAD", "geometric robustness exact arithmetic" |
| Healing / repair | Fixing imported STEP data | "shape healing CAD", "STEP data repair" |
| Validity checking | Checking that a solid is well-formed | "B-Rep validity check", "solid validation self-intersection" |
| Tessellation | Converting to triangles for display and sampling | "B-Rep tessellation", "surface triangulation chordal deviation" |
| Mass properties | Volume, area, centroid | "divergence theorem volume computation", "Gauss quadrature surface integral" |
| Point-in-solid classification | Monte Carlo IoU, inside/outside tests | "ray casting point in solid", "point membership classification" |
| History replay and regeneration | Re-executing the feature tree | "parametric model regeneration", "history-based vs direct modeling" |

**Study prompt:**
> Explain how a commercial or open-source geometric modelling kernel (Parasolid, ACIS, Open
> CASCADE, CATIA's CGM) works internally. Cover the separation of geometry and topology, how an
> extrusion and a Boolean subtraction are computed (surface–surface intersection, splitting and
> classifying faces, rebuilding topology), how constant-radius fillets are computed (rolling-ball
> blends), and what "tolerant modelling" means. Explain why near-coincident faces and tiny
> "sliver" edges cause failures, how healing works, and how volume is computed via the divergence
> theorem. Explain what happens when a feature tree is regenerated after a parameter change.

---

## Part N — Learning on CAD: the research landscape

| Topic | Representative works (search by name) |
|---|---|
| Learning directly on B-Reps | UV-Net (Jayaraman et al., 2021), BRepNet (Lambourne et al., 2021), BRT / "Bringing attention to CAD" (Zou & Zhu, 2025) |
| Face-level segmentation and feature recognition | "MFCAD dataset", "Hierarchical CADNet", "AAGNet" |
| Generating CAD construction sequences | DeepCAD (Wu et al., 2021), SkexGen (Xu et al., 2022), HNC-CAD (Xu et al., 2023) |
| Reverse engineering into CAD programs | CAD-SIGNet (2024), CAD-Recode (2024/2025), Point2Cyl, "Fusion 360 Gallery reconstruction" |
| Generating B-Reps directly | SolidGen (2022), BrepGen (2024) |
| Datasets | ABC dataset, Fusion 360 Gallery dataset, DeepCAD dataset, TMCAD |
| Inverse CSG and shape programs | CSGNet, "ShapeAssembly", "neural shape programs" |
| Geometry-informed learning | "Geometry-Informed Neural Networks (GINNs)", "physics-informed neural networks (PINNs)", "differentiable CAD", "signed distance functions" |

**Study prompt:**
> Give me a literature map of deep learning for CAD reverse engineering, in chronological order:
> B-Rep encoders (UV-Net, BRepNet, B-Rep Transformers), CAD sequence generators (DeepCAD,
> SkexGen, HNC-CAD), reconstruction from point clouds or B-Reps into CAD programs (CAD-SIGNet,
> CAD-Recode, Fusion 360 Gallery reconstruction), and direct B-Rep generation (SolidGen,
> BrepGen). For each, summarise the input representation, the output representation, the
> architecture and the main limitation. Then explain why PINNs do not directly apply to
> CAD-history recovery, and which "geometry-informed" alternatives (execution-based rewards,
> differentiable SDF proxies) are used instead.

---

## Recommended study order

1. **Part A** (maths), then **Part H** (B-Rep), then **Part I** (curves and surfaces), then **Part K** (UV space).
   After these you understand the *input data*.
2. **Part B** (neural networks), then **Part C** (CNNs), then **Part D** (GNNs). After these you understand the *encoder*.
3. **Part E** (Transformers), then **Part F** (encoder–decoder), then **Part G** (decoding and evaluation).
   After these you understand the *decoder and output*.
4. **Parts J, L, M** (differential geometry, continuity, kernel internals). After these you
   understand why rebuilds succeed or fail, and how the predicted history behaves in the CAD system.
5. **Part N** (research landscape). After this you can place this system among published work
   and choose next steps.

## Core references (search by title)

- *The NURBS Book* — Piegl & Tiller
- *Curves and Surfaces for CAGD* — Farin
- *Geometric Modeling* — Mortenson
- *An Introduction to Solid Modeling* — Mäntylä
- *Differential Geometry of Curves and Surfaces* — do Carmo
- *Deep Learning* — Goodfellow, Bengio & Courville
- *Dive into Deep Learning* — Zhang et al. (free online)
- *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* — Bronstein et al.
- *Graph Representation Learning* — Hamilton
- *Attention Is All You Need* — Vaswani et al., 2017
- *UV-Net: Learning from Boundary Representations* — Jayaraman et al., CVPR 2021
- *DeepCAD: A Deep Generative Network for Computer-Aided Design Models* — Wu et al., ICCV 2021
