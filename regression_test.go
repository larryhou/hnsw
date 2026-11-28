package hnsw

// Regression tests for the behaviours this fork changed relative to upstream
// coder/hnsw v0.6.1. Each test names the change it protects; if one fails,
// read the comment before "fixing" the test.

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

// ---- helpers ---------------------------------------------------------------

func seqKeys(n int) []string {
	keys := make([]string, n)
	for i := range keys {
		keys[i] = fmt.Sprintf("k%04d", i)
	}
	return keys
}

// clusteredVectors returns n unit vectors spread over `clusters` tight
// clusters. Clustered data is where naive "keep the M nearest" neighbour
// selection loses connectivity between clusters.
func clusteredVectors(n, dim, clusters int, spread float64, seed int64) []Vector {
	r := rand.New(rand.NewSource(seed))
	centers := randomVectors(clusters, dim, seed+99)
	out := make([]Vector, n)
	for i := range out {
		c := centers[i%clusters]
		v := make(Vector, dim)
		var sum float64
		for j := range v {
			x := float64(c[j]) + r.NormFloat64()*spread
			v[j] = float32(x)
			sum += x * x
		}
		nm := float32(1 / math.Sqrt(sum))
		for j := range v {
			v[j] *= nm
		}
		out[i] = v
	}
	return out
}

func buildGraph(keys []string, vecs []Vector, mutate func(*Graph[string])) *Graph[string] {
	g := NewGraph[string]()
	if mutate != nil {
		mutate(g)
	}
	for i := range keys {
		g.Add(MakeNode(keys[i], vecs[i]))
	}
	return g
}

func recallAt(g *Graph[string], keys []string, vecs, queries []Vector, k int) float64 {
	hit, total := 0, 0
	for _, q := range queries {
		want := map[string]bool{}
		for _, w := range bruteTopK(keys, vecs, q, k) {
			want[w] = true
		}
		for _, got := range g.Search(q, k) {
			if want[got.Key] {
				hit++
			}
		}
		total += k
	}
	return float64(hit) / float64(total)
}

func selfMisses(g *Graph[string], keys []string, vecs []Vector) int {
	miss := 0
	for i := range keys {
		r := g.Search(vecs[i], 1)
		if len(r) == 0 || r[0].Key != keys[i] {
			miss++
		}
	}
	return miss
}

// ---- search correctness ----------------------------------------------------

// Protects: search termination. Upstream v0.6.1 stopped as soon as one
// expansion round failed to improve the best distance (hill climbing), so a
// graph over a few hundred vectors found only ~50% of true neighbours and
// raising EfSearch did nothing. Raising EfSearch must monotonically (within
// noise) improve recall, and reach near-exact results.
func TestRegression_EfSearchControlsRecall(t *testing.T) {
	const n, dim, k = 500, 128, 5
	vecs := randomVectors(n, dim, 11)
	keys := seqKeys(n)
	queries := randomVectors(100, dim, 12)

	recall := func(ef int) float64 {
		g := buildGraph(keys, vecs, func(g *Graph[string]) { g.EfSearch = ef })
		return recallAt(g, keys, vecs, queries, k)
	}
	low, high := recall(10), recall(150)
	require.Greater(t, high, low+0.05, "EfSearch must matter: ef=10 -> %.3f, ef=150 -> %.3f", low, high)
	require.GreaterOrEqual(t, high, 0.98, "ef=150 recall@%d = %.3f", k, high)
}

// Protects: search result ordering. Upstream returned the raw heap slice,
// which is not sorted. Callers (and the Search contract) expect ascending
// distance.
func TestRegression_SearchResultsSortedByDistance(t *testing.T) {
	const n, dim = 300, 32
	vecs := randomVectors(n, dim, 21)
	g := buildGraph(seqKeys(n), vecs, nil)
	for _, q := range randomVectors(30, dim, 22) {
		res := g.SearchWithDistance(q, 10)
		require.Len(t, res, 10)
		for i := 1; i < len(res); i++ {
			require.LessOrEqual(t, res[i-1].Distance, res[i].Distance, "result %d out of order", i)
		}
	}
}

// Protects: exactly k results are returned when the graph is large enough,
// even when k > EfSearch (the search budget must be raised to k).
func TestRegression_SearchReturnsKEvenWhenKExceedsEf(t *testing.T) {
	const n, dim = 200, 16
	vecs := randomVectors(n, dim, 31)
	g := buildGraph(seqKeys(n), vecs, func(g *Graph[string]) { g.EfSearch = 5 })
	res := g.Search(vecs[0], 40)
	require.Len(t, res, 40)
	seen := map[string]bool{}
	for _, r := range res {
		require.False(t, seen[r.Key], "duplicate key %s in results", r.Key)
		seen[r.Key] = true
	}
}

// Protects: Search(k) must return min(k, Len()) results on tiny graphs.
func TestRegression_SearchOnTinyGraph(t *testing.T) {
	g := NewGraph[string]()
	require.Empty(t, g.Search(Vector{1, 0}, 3))
	g.Add(MakeNode("a", Vector{1, 0}))
	require.Len(t, g.Search(Vector{1, 0}, 3), 1)
	g.Add(MakeNode("b", Vector{0, 1}))
	g.Add(MakeNode("c", Vector{1, 1}))
	res := g.Search(Vector{1, 0}, 10)
	require.Len(t, res, 3)
	require.Equal(t, "a", res[0].Key)
}

// ---- determinism -----------------------------------------------------------

// Protects: build determinism. Upstream seeded the RNG with time.Now() and
// picked the layer entry node from a map, so two builds of the same input
// answered the same query differently. The internal graph structure (not just
// search output) must be identical across builds.
func TestRegression_BuildIsStructurallyDeterministic(t *testing.T) {
	const n, dim = 300, 48
	vecs := clusteredVectors(n, dim, 12, 0.02, 41)
	keys := seqKeys(n)

	signature := func(g *Graph[string]) string {
		var out []string
		for li, l := range g.layers {
			ks := make([]string, 0, len(l.nodes))
			for k := range l.nodes {
				ks = append(ks, k)
			}
			slices.Sort(ks)
			for _, k := range ks {
				nbrs := make([]string, 0)
				for nk := range l.nodes[k].neighbors {
					nbrs = append(nbrs, nk)
				}
				slices.Sort(nbrs)
				out = append(out, fmt.Sprintf("L%d %s -> %v", li, k, nbrs))
			}
		}
		return fmt.Sprint(out)
	}

	base := signature(buildGraph(keys, vecs, nil))
	for i := 0; i < 5; i++ {
		require.Equal(t, base, signature(buildGraph(keys, vecs, nil)), "build %d differs", i)
	}
}

// Protects: NewGraph must not seed from the clock.
func TestRegression_NewGraphUsesFixedSeed(t *testing.T) {
	a, b := NewGraph[string](), NewGraph[string]()
	for i := 0; i < 100; i++ {
		require.Equal(t, a.Rng.Float64(), b.Rng.Float64(), "draw %d", i)
	}
}

// Protects: stability of search output when the same nodes are inserted in
// the same order, including ties. Equal-distance candidates must be ordered
// by key, never by map iteration order.
func TestRegression_TiesBrokenByKey(t *testing.T) {
	// Four points at identical distance from the query.
	g := NewGraph[string]()
	g.Distance = EuclideanDistance
	for _, k := range []string{"d", "b", "a", "c"} {
		g.Add(MakeNode(k, Vector{1, 0}))
	}
	g.Add(MakeNode("far", Vector{10, 10}))
	for i := 0; i < 50; i++ {
		res := g.Search(Vector{1, 0}, 4)
		got := make([]string, len(res))
		for j, r := range res {
			got[j] = r.Key
		}
		require.Equal(t, []string{"a", "b", "c", "d"}, got, "iteration %d", i)
	}
}

// Protects: the layer entry point is the smallest key regardless of map order.
func TestRegression_EntryIsSmallestKeyAcrossLayers(t *testing.T) {
	g := NewGraph[string]()
	vecs := randomVectors(200, 8, 51)
	keys := seqKeys(200)
	for i := range keys {
		g.Add(MakeNode(keys[i], vecs[i]))
	}
	for li, l := range g.layers {
		want := ""
		for k := range l.nodes {
			if want == "" || k < want {
				want = k
			}
		}
		for i := 0; i < 20; i++ {
			require.Equal(t, want, l.entry().Key, "layer %d", li)
		}
	}
}

// ---- neighbour selection heuristic ----------------------------------------

// Protects: selectNeighbors unit behaviour (HNSW paper, Algorithm 4).
// Geometry on a line, base point at 0:
//
//	a=1, b=1.1 (shadowed by a, since |a-b| < |b-base|), c=-2, d=3 (shadowed by a)
//
// With m=2 the heuristic keeps the nearest (a) and the nearest candidate that
// is not shadowed (c, on the other side), instead of the two nearest (a, b).
func TestRegression_SelectNeighborsPrefersDiverseDirections(t *testing.T) {
	mk := func(key string, x float32) searchCandidate[string] {
		return searchCandidate[string]{
			node: &layerNode[string]{Node: Node[string]{Key: key, Value: Vector{x}}},
			dist: float32(math.Abs(float64(x))),
		}
	}
	cands := []searchCandidate[string]{mk("a", 1), mk("b", 1.1), mk("c", -2), mk("d", 3)}
	sortCandidates(cands)

	got := selectNeighbors(cands, 2, EuclideanDistance)
	keys := []string{got[0].node.Key, got[1].node.Key}
	require.Equal(t, []string{"a", "c"}, keys)
}

// Protects: the heuristic tops the set up to m from pruned candidates
// (keepPrunedConnections), so nodes are never left under-connected.
func TestRegression_SelectNeighborsFillsToM(t *testing.T) {
	mk := func(key string, x float32) searchCandidate[string] {
		return searchCandidate[string]{
			node: &layerNode[string]{Node: Node[string]{Key: key, Value: Vector{x}}},
			dist: x,
		}
	}
	// All candidates are collinear on one side: all but the first are shadowed.
	cands := []searchCandidate[string]{mk("a", 1), mk("b", 2), mk("c", 3), mk("d", 4)}
	got := selectNeighbors(cands, 3, EuclideanDistance)
	require.Len(t, got, 3)
	require.Equal(t, "a", got[0].node.Key)
	// Pruned candidates fill the rest in distance order.
	require.Equal(t, "b", got[1].node.Key)
	require.Equal(t, "c", got[2].node.Key)
}

func TestRegression_SelectNeighborsPassThroughWhenFewCandidates(t *testing.T) {
	c := []searchCandidate[string]{{node: &layerNode[string]{Node: Node[string]{Key: "x", Value: Vector{1}}}, dist: 1}}
	require.Equal(t, c, selectNeighbors(c, 4, EuclideanDistance))
	require.Empty(t, selectNeighbors[string](nil, 4, EuclideanDistance))
}

// Protects: no node ever exceeds M neighbours and has no self loop, under both
// policies.
func TestRegression_DegreeBound(t *testing.T) {
	const n, dim = 400, 32
	vecs := clusteredVectors(n, dim, 20, 0.02, 61)
	keys := seqKeys(n)
	for _, simple := range []bool{false, true} {
		g := buildGraph(keys, vecs, func(g *Graph[string]) { g.SimpleNeighbors = simple })
		for li, l := range g.layers {
			for k, node := range l.nodes {
				require.LessOrEqual(t, len(node.neighbors), g.M, "simple=%v layer %d node %s degree", simple, li, k)
				require.NotContains(t, node.neighbors, k, "self loop at %s", k)
			}
		}
	}
}

// layer0Reachable counts nodes reachable from the layer-0 entry by following
// directed edges.
func layer0Reachable(g *Graph[string]) int {
	l0 := g.layers[0]
	start := l0.entry()
	seen := map[string]bool{start.Key: true}
	stack := []*layerNode[string]{start}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for k, nb := range cur.neighbors {
			if !seen[k] {
				seen[k] = true
				stack = append(stack, nb)
			}
		}
	}
	return len(seen)
}

// Protects: graph connectivity on clustered data. This is the main effect of
// the neighbour-selection heuristic. Measured reference: with the simple
// "keep the M nearest" policy only ONE cluster (20 of 400 nodes) is reachable
// from the entry node, because every node's edges stay inside its own
// cluster; with the heuristic all nodes are reachable. A failure here means
// the heuristic is no longer applied on insertion (check Add, SimpleNeighbors
// default, selectNeighbors).
func TestRegression_HeuristicKeepsClustersConnected(t *testing.T) {
	for _, tc := range []struct {
		n, dim, clusters int
		spread           float64
	}{
		{400, 32, 20, 0.02},
		{600, 128, 30, 0.01},
		{1000, 256, 50, 0.01},
	} {
		vecs := clusteredVectors(tc.n, tc.dim, tc.clusters, tc.spread, 61)
		g := buildGraph(seqKeys(tc.n), vecs, nil)
		require.Equal(t, tc.n, layer0Reachable(g), "n=%d dim=%d clusters=%d", tc.n, tc.dim, tc.clusters)
	}
}

// Protects: the heuristic's reason for existing. On clustered data it must
// beat (or at least match) the simple policy and stay near-exact. Measured
// reference (1000 vecs, d=256, 50 clusters, ef=100): simple 0.867, heuristic
// 0.989. If this fails, the heuristic is probably no longer being applied
// (e.g. SimpleNeighbors default flipped, or selectNeighbors bypassed in Add).
func TestRegression_HeuristicBeatsSimpleOnClusters(t *testing.T) {
	const n, dim, k = 1000, 256, 5
	vecs := clusteredVectors(n, dim, 50, 0.01, 71)
	keys := seqKeys(n)
	queries := clusteredVectors(200, dim, 10, 0.02, 72)

	build := func(simple bool) *Graph[string] {
		return buildGraph(keys, vecs, func(g *Graph[string]) {
			g.EfSearch = 100
			g.SimpleNeighbors = simple
		})
	}
	simple := recallAt(build(true), keys, vecs, queries, k)
	heur := recallAt(build(false), keys, vecs, queries, k)
	require.Greater(t, heur, simple+0.05, "heuristic %.3f vs simple %.3f", heur, simple)
	require.GreaterOrEqual(t, heur, 0.97, "heuristic recall@%d = %.3f", k, heur)
}

// Protects: the default policy. NewGraph must use the heuristic; flipping the
// default silently would reintroduce the cluster recall loss.
func TestRegression_HeuristicIsDefault(t *testing.T) {
	require.False(t, NewGraph[string]().SimpleNeighbors)
}

// Protects: every stored vector finds itself with the default parameters on
// clustered data (the failure mode seen in production: documents that could
// not be found by their own text).
func TestRegression_StoredVectorsFindThemselves(t *testing.T) {
	const n, dim = 600, 128
	vecs := clusteredVectors(n, dim, 30, 0.01, 81)
	keys := seqKeys(n)
	g := buildGraph(keys, vecs, func(g *Graph[string]) { g.EfSearch = 100 })
	require.Equal(t, 0, selfMisses(g, keys, vecs))
}

// ---- mutation --------------------------------------------------------------

// Protects: Add with an existing key replaces the node and leaves exactly one
// copy, found at its new position.
func TestRegression_AddExistingKeyReplaces(t *testing.T) {
	g := NewGraph[string]()
	g.Distance = EuclideanDistance
	for i := 0; i < 30; i++ {
		g.Add(MakeNode(fmt.Sprintf("k%02d", i), Vector{float32(i), 0}))
	}
	g.Add(MakeNode("k05", Vector{100, 100}))
	require.Equal(t, 30, g.Len())
	v, ok := g.Lookup("k05")
	require.True(t, ok)
	require.Equal(t, Vector{100, 100}, v)
	res := g.Search(Vector{100, 100}, 1)
	require.Equal(t, "k05", res[0].Key)
}

// Protects: Delete keeps the graph searchable and degree-bounded under both
// policies, and uses the graph's own distance function.
func TestRegression_DeleteKeepsGraphUsable(t *testing.T) {
	const n, dim = 200, 16
	vecs := randomVectors(n, dim, 91)
	keys := seqKeys(n)
	for _, simple := range []bool{false, true} {
		g := buildGraph(keys, vecs, func(g *Graph[string]) { g.SimpleNeighbors = simple; g.EfSearch = 60 })
		for i := 0; i < n; i += 3 {
			require.True(t, g.Delete(keys[i]))
		}
		require.False(t, g.Delete("missing"))
		for _, l := range g.layers {
			for _, node := range l.nodes {
				require.LessOrEqual(t, len(node.neighbors), g.M)
				for nk := range node.neighbors {
					_, alive := l.nodes[nk]
					require.True(t, alive, "dangling edge to deleted node %s", nk)
				}
			}
		}
		for i := 1; i < n; i += 3 {
			res := g.Search(vecs[i], 1)
			require.NotEmpty(t, res)
			require.Equal(t, keys[i], res[0].Key, "simple=%v", simple)
		}
	}
}

// Protects: Export/Import round-trips a graph built with the new defaults.
func TestRegression_ExportImportRoundTrip(t *testing.T) {
	const n, dim = 150, 24
	vecs := randomVectors(n, dim, 101)
	keys := seqKeys(n)
	g := buildGraph(keys, vecs, nil)

	var buf bytesBuffer
	require.NoError(t, g.Export(&buf))
	g2 := NewGraph[string]()
	require.NoError(t, g2.Import(&buf))
	require.Equal(t, g.Len(), g2.Len())
	for _, q := range randomVectors(20, dim, 102) {
		a, b := g.Search(q, 5), g2.Search(q, 5)
		require.Equal(t, len(a), len(b))
		for i := range a {
			require.Equal(t, a[i].Key, b[i].Key)
		}
	}
}

type bytesBuffer = bytes.Buffer

// Protects: worstNeighbor tie-breaking. When several neighbours are at exactly
// the same distance (duplicate or symmetric vectors), the one dropped must be
// chosen by key, not by map iteration order, otherwise graph structure (and
// search output) changes from build to build.
func TestRegression_WorstNeighborTieBrokenByKey(t *testing.T) {
	n := &layerNode[string]{Node: Node[string]{Key: "self", Value: Vector{0, 0}}, neighbors: map[string]*layerNode[string]{}}
	for _, k := range []string{"m", "c", "x", "a", "q", "t", "b"} {
		// All at distance 1 from n.
		n.neighbors[k] = &layerNode[string]{Node: Node[string]{Key: k, Value: Vector{1, 0}}}
	}
	for i := 0; i < 100; i++ {
		require.Equal(t, "x", n.worstNeighbor(EuclideanDistance).Key, "iteration %d", i)
	}
}

// Protects: the same tie-breaking end to end through addNeighbor with the
// simple policy (overflow drops the worst neighbour).
func TestRegression_AddNeighborOverflowTieDeterministic(t *testing.T) {
	const m = 3
	for i := 0; i < 50; i++ {
		n := &layerNode[string]{Node: Node[string]{Key: "self", Value: Vector{0, 0}}}
		for _, k := range []string{"d", "b", "a", "c"} {
			n.addNeighbor(&layerNode[string]{Node: Node[string]{Key: k, Value: Vector{1, 0}}}, m, EuclideanDistance, true)
		}
		got := make([]string, 0, m)
		for k := range n.neighbors {
			got = append(got, k)
		}
		slices.Sort(got)
		require.Equal(t, []string{"a", "b", "c"}, got, "iteration %d", i)
	}
}

// Protects: search termination. The search must stop once the closest
// unexplored candidate is farther than the worst result (HNSW Algorithm 2),
// i.e. it must NOT degenerate into a full scan, and it must not stop early on
// a local optimum either. Distance evaluations are counted: a healthy search
// touches a small fraction of the graph, while still reaching high recall.
func TestRegression_SearchTerminatesEarlyButFindsNeighbors(t *testing.T) {
	const n, dim, k = 2000, 64, 5
	vecs := clusteredVectors(n, dim, 40, 0.03, 111)
	keys := seqKeys(n)
	queries := clusteredVectors(100, dim, 10, 0.03, 112)

	var calls int
	counting := func(a, b []float32) float32 { calls++; return CosineDistance(a, b) }
	g := buildGraph(keys, vecs, func(g *Graph[string]) { g.EfSearch = 100 })
	g.Distance = counting

	calls = 0
	hit, total := 0, 0
	for _, q := range queries {
		want := map[string]bool{}
		for _, w := range bruteTopK(keys, vecs, q, k) {
			want[w] = true
		}
		for _, got := range g.Search(q, k) {
			if want[got.Key] {
				hit++
			}
		}
		total += k
	}
	perQuery := float64(calls) / float64(len(queries))
	recall := float64(hit) / float64(total)

	// Measured reference (ef=100): ~766 distance evaluations/query (38% of
	// the graph) at recall 0.998. Never terminating early costs ~1313 (66%);
	// terminating immediately gives recall 0. Thresholds sit between them.
	require.Less(t, perQuery, float64(n)*0.5, "search touched %.0f of %d nodes per query: termination too late (scanning too much)", perQuery, n)
	require.GreaterOrEqual(t, recall, 0.98, "recall@%d = %.3f: termination too early", k, recall)
}

// Protects: a larger EfSearch must never do LESS work than a smaller one, and
// must explore strictly more on a non-trivial graph. Catches ef being ignored.
func TestRegression_EfSearchIncreasesExploration(t *testing.T) {
	const n, dim = 1500, 64
	vecs := randomVectors(n, dim, 121)
	keys := seqKeys(n)
	queries := randomVectors(40, dim, 122)

	work := func(ef int) float64 {
		var calls int
		g := buildGraph(keys, vecs, func(g *Graph[string]) { g.EfSearch = ef })
		g.Distance = func(a, b []float32) float32 { calls++; return CosineDistance(a, b) }
		calls = 0
		for _, q := range queries {
			g.Search(q, 5)
		}
		return float64(calls) / float64(len(queries))
	}
	small, large := work(10), work(200)
	require.Greater(t, large, small*1.5, "ef=200 explored %.0f nodes/query vs ef=10 %.0f", large, small)
}

// Protects: the layer entry point does not depend on map iteration order.
// A hand-built graph whose top layer holds several mutually unconnected nodes
// makes the entry choice observable: each isolated node can only return
// itself, so Search output reveals which one was picked. It must be the
// smallest key, on every call.
func TestRegression_SearchEntryIsDeterministic(t *testing.T) {
	mkNode := func(k string, x float32) *layerNode[string] {
		return &layerNode[string]{Node: Node[string]{Key: k, Value: Vector{x, 0}}}
	}
	g := NewGraph[string]()
	g.Distance = EuclideanDistance
	nodes := map[string]*layerNode[string]{}
	for i, k := range []string{"m", "c", "x", "a", "q", "t", "b"} {
		nodes[k] = mkNode(k, float32(i*10))
	}
	// Single layer, no edges at all: Search can only return the entry node.
	g.layers = []*layer[string]{{nodes: nodes}}

	for i := 0; i < 100; i++ {
		res := g.Search(Vector{1000, 0}, 1)
		require.Len(t, res, 1)
		require.Equal(t, "a", res[0].Key, "call %d: entry must be the smallest key", i)
	}
}

// Protects: re-adding an existing key (an in-place update) must not corrupt
// the graph. Add used to delete the old node one layer at a time while
// inserting, which also removed the lower layers' copies; the elevator then
// pointed at a node that was gone and a later Add panicked with a nil
// searchPoint. The panic needs several updates and enough layers to show up,
// so this runs many of them over a range of seeds.
func TestRegression_RepeatedInPlaceUpdates(t *testing.T) {
	const n, dim, updates = 120, 8, 240
	for seed := int64(1); seed <= 30; seed++ {
		r := rand.New(rand.NewSource(seed))
		vecs := randomVectors(n, dim, seed)
		g := NewGraph[int]()
		g.Rng = rand.New(rand.NewSource(seed))
		for i := 0; i < n; i++ {
			g.Add(MakeNode(i, vecs[i]))
		}
		latest := map[int]Vector{}
		for i := 0; i < n; i++ {
			latest[i] = vecs[i]
		}
		for u := 0; u < updates; u++ {
			k := r.Intn(n)
			v := make(Vector, dim)
			for d := range v {
				v[d] = vecs[k][d] + float32(r.NormFloat64())*0.1
			}
			require.NotPanics(t, func() { g.Add(MakeNode(k, v)) }, "seed %d update %d key %d", seed, u, k)
			latest[k] = v
		}

		require.Equal(t, n, g.Len(), "seed %d", seed)
		for k, want := range latest {
			got, ok := g.Lookup(k)
			require.True(t, ok, "seed %d key %d missing", seed, k)
			require.Equal(t, want, got, "seed %d key %d holds a stale vector", seed, k)
		}
		for li, l := range g.layers {
			require.NotEmpty(t, l.nodes, "seed %d layer %d is empty", seed, li)
			for k, node := range l.nodes {
				if li > 0 {
					_, below := g.layers[li-1].nodes[k]
					require.True(t, below, "seed %d: key %d in layer %d but not below", seed, k, li)
				}
				for nk, nb := range node.neighbors {
					require.Same(t, l.nodes[nk], nb, "seed %d layer %d: edge %d->%d points at a removed node", seed, li, k, nk)
				}
			}
		}
	}
}

// Protects: updating a key in place keeps recall close to a graph built from
// scratch from the same data. Before replaceEfFactor, re-adding a key searched
// only EfSearch wide, and after several rounds of updates recall at a given
// EfSearch sat 4-7 points below a fresh build.
func TestRegression_RecallAfterInPlaceUpdates(t *testing.T) {
	const n, dim, k, ef, rounds = 1000, 64, 10, 20, 8
	r := rand.New(rand.NewSource(3))
	vecs := randomVectors(n, dim, 3)
	keys := seqKeys(n)

	g := NewGraph[string]()
	g.EfSearch = ef
	cur := append([]Vector(nil), vecs...)
	for i := range keys {
		g.Add(MakeNode(keys[i], cur[i]))
	}
	for round := 0; round < rounds; round++ {
		for i := range keys {
			if r.Float64() >= 0.3 {
				continue
			}
			v := make(Vector, dim)
			for d := range v {
				v[d] = cur[i][d] + float32(r.NormFloat64())*0.08
			}
			cur[i] = v
			g.Add(MakeNode(keys[i], v))
		}
	}
	require.Equal(t, n, g.Len())

	fresh := buildGraph(keys, cur, func(g *Graph[string]) { g.EfSearch = ef })
	queries := randomVectors(150, dim, 503)
	updated := recallAt(g, keys, cur, queries, k)
	base := recallAt(fresh, keys, cur, queries, k)
	t.Logf("recall@%d updated=%.3f fresh=%.3f", k, updated, base)
	require.GreaterOrEqual(t, updated, base-0.03,
		"recall after in-place updates %.3f is far below a fresh build %.3f", updated, base)
}
