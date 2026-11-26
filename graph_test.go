package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_maxLevel(t *testing.T) {
	var m int

	m = maxLevel(0.5, 10)
	require.Equal(t, 4, m)

	m = maxLevel(0.5, 1000)
	require.Equal(t, 11, m)
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[int]{
		Node: Node[int]{
			Value: Vector{0},
			Key:   0,
		},
		neighbors: map[int]*layerNode[int]{
			1: {
				Node: Node[int]{
					Value: Vector{1},
					Key:   1,
				},
			},
			2: {
				Node: Node[int]{
					Value: Vector{2},
					Key:   2,
				},
			},
			3: {
				Node: Node[int]{
					Value: Vector{3},
					Key:   3,
				},
				neighbors: map[int]*layerNode[int]{
					4: {
						Node: Node[int]{
							Value: Vector{4},
							Key:   5,
						},
					},
					5: {
						Node: Node[int]{
							Value: Vector{5},
							Key:   5,
						},
					},
				},
			},
		},
	}

	best := entry.search(2, 4, []float32{4}, EuclideanDistance)

	require.Equal(t, 5, best[0].node.Key)
	require.Equal(t, 3, best[1].node.Key)
	require.Len(t, best, 2)
}

func newTestGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        6,
		Distance: EuclideanDistance,
		Ml:       0.5,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(0)),
	}
}

func TestGraph_AddSearch(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()

	for i := 0; i < 128; i++ {
		g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
	}

	al := Analyzer[int]{Graph: g}

	// Layers should be approximately log2(128) = 7
	// Look for an approximate doubling of the number of nodes in each layer.
	require.Equal(t, []int{
		128,
		67,
		28,
		12,
		6,
		2,
		1,
		1,
	}, al.Topography())

	nearest := g.Search(
		[]float32{64.5},
		4,
	)

	require.Len(t, nearest, 4)
	// The two closest are 64 and 65 (distance 0.5 each).
	// The next two are 63 and 66 (distance 1.5 each).
	require.EqualValues(
		t,
		[]Node[int]{
			{64, Vector{64}},
			{65, Vector{65}},
			{63, Vector{63}},
			{66, Vector{66}},
		},
		nearest,
	)
}

func TestGraph_AddDelete(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g.Add(Node[int]{
			Key:   i,
			Value: Vector{float32(i)},
		})
	}

	require.Equal(t, 128, g.Len())
	an := Analyzer[int]{Graph: g}

	preDeleteConnectivity := an.Connectivity()

	// Delete every even node.
	for i := 0; i < 128; i += 2 {
		ok := g.Delete(i)
		require.True(t, ok)
	}

	require.Equal(t, 64, g.Len())

	postDeleteConnectivity := an.Connectivity()

	// Connectivity should be (almost) the same for the lowest layer. It used
	// to be asserted exactly equal, but that only held because edges to
	// deleted nodes were left dangling and still counted towards the degree.
	// Delete now purges them, so a small drop is expected and correct.
	require.InDelta(
		t, preDeleteConnectivity[0],
		postDeleteConnectivity[0],
		0.5,
	)

	// No remaining node may point at a deleted node.
	for li, l := range g.layers {
		for k, node := range l.nodes {
			for nk := range node.neighbors {
				_, alive := l.nodes[nk]
				require.True(t, alive, "layer %d: node %v has dangling edge to %v", li, k, nk)
			}
		}
	}

	t.Run("DeleteNotFound", func(t *testing.T) {
		ok := g.Delete(-1)
		require.False(t, ok)
	})
}

func Benchmark_HSNW(b *testing.B) {
	b.ReportAllocs()

	sizes := []int{100, 1000, 10000}

	// Use this to ensure that complexity is O(log n) where n = h.Len().
	for _, size := range sizes {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			g := Graph[int]{}
			g.Ml = 0.5
			g.Distance = EuclideanDistance
			for i := 0; i < size; i++ {
				g.Add(Node[int]{
					Key:   i,
					Value: Vector{float32(i)},
				})
			}
			b.ResetTimer()

			b.Run("Search", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					g.Search(
						[]float32{float32(i % size)},
						4,
					)
				}
			})
		})
	}
}

func randFloats(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rand.Float32()
	}
	return x
}

func Benchmark_HNSW_1536(b *testing.B) {
	b.ReportAllocs()

	g := newTestGraph[int]()
	const size = 1000
	points := make([]Node[int], size)
	for i := 0; i < size; i++ {
		points[i] = Node[int]{
			Key:   i,
			Value: Vector(randFloats(1536)),
		}
		g.Add(points[i])
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g.Search(
				points[i%size].Value,
				4,
			)
		}
	})
}

func TestGraph_DefaultCosine(t *testing.T) {
	g := NewGraph[int]()
	g.Add(
		Node[int]{Key: 1, Value: Vector{1, 1}},
		Node[int]{Key: 2, Value: Vector{0, 1}},
		Node[int]{Key: 3, Value: Vector{1, -1}},
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5},
		1,
	)

	require.Equal(
		t,
		[]Node[int]{
			{1, Vector{1, 1}},
		},
		neighbors,
	)
}

func TestGraph_RemoveAllNodes(t *testing.T) {
	var vec = []float32{1}

	for i := 0; i < 10; i++ {
		g := NewGraph[int]()
		g.Add(MakeNode(1, vec))
		g.Delete(1)
		g.Add(MakeNode(1, vec))
	}
}

func TestGraph_SearchFindsCorrectNearest(t *testing.T) {
	// With the old search algorithm, termination was too aggressive (stopped
	// when no neighbor beat result.Min) and the result set was bounded by k
	// instead of efSearch. This caused the search to miss true nearest
	// neighbors, especially with small k and large efSearch.
	g := newTestGraph[int]()
	for i := 0; i < 100; i++ {
		g.Add(Node[int]{Key: i, Value: Vector{float32(i)}})
	}

	// Search for k=1 nearest to 50.5. The answer must be 50 or 51.
	results := g.Search(Vector{50.5}, 1)
	require.Len(t, results, 1)
	require.Contains(t, []int{50, 51}, results[0].Key,
		"expected nearest neighbor to 50.5, got key=%d", results[0].Key)

	// Search for k=3 nearest to 0.0. Must return 0, 1, 2.
	results = g.Search(Vector{0.0}, 3)
	require.Len(t, results, 3)
	keys := make([]int, 3)
	for i, r := range results {
		keys[i] = r.Key
	}
	require.Subset(t, []int{0, 1, 2}, keys)
}

func TestGraph_DeleteReplenishUsesGraphDistance(t *testing.T) {
	// replenish() previously hardcoded CosineDistance. After deleting a
	// node from a EuclideanDistance graph, replenish must use the correct
	// distance function or the topology becomes corrupted.
	g := newTestGraph[int]() // uses EuclideanDistance
	for i := 0; i < 20; i++ {
		g.Add(Node[int]{Key: i, Value: Vector{float32(i)}})
	}

	// Delete a node in the middle to trigger replenish.
	g.Delete(10)

	// Search should still find the correct nearest neighbor.
	results := g.Search(Vector{9.5}, 1)
	require.Len(t, results, 1)
	// Must be 9 or 11 (both distance 0.5 from 9.5).
	require.Contains(t, []int{9, 11}, results[0].Key)
}

func randomVectors(n, dim int, seed int64) []Vector {
	r := rand.New(rand.NewSource(seed))
	out := make([]Vector, n)
	for i := range out {
		v := make(Vector, dim)
		var sum float64
		for j := range v {
			x := r.NormFloat64()
			v[j] = float32(x)
			sum += x * x
		}
		norm := float32(1 / math.Sqrt(sum))
		for j := range v {
			v[j] *= norm
		}
		out[i] = v
	}
	return out
}

func bruteTopK(keys []string, vecs []Vector, q Vector, k int) []string {
	idx := make([]int, len(keys))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int {
		return cmp.Compare(CosineDistance(vecs[a], q), CosineDistance(vecs[b], q))
	})
	out := make([]string, k)
	for i := range out {
		out[i] = keys[idx[i]]
	}
	return out
}

// Building twice from the same input must produce identical search results,
// regardless of Go map iteration order.
func TestGraph_Deterministic(t *testing.T) {
	const n, dim = 400, 64
	vecs := randomVectors(n, dim, 1)
	keys := make([]string, n)
	for i := range keys {
		keys[i] = fmt.Sprintf("k%04d", i)
	}
	build := func() *Graph[string] {
		g := NewGraph[string]()
		for i := range keys {
			g.Add(MakeNode(keys[i], vecs[i]))
		}
		return g
	}
	queries := randomVectors(50, dim, 2)
	first := build()
	for round := 0; round < 5; round++ {
		g := build()
		for qi, q := range queries {
			a, b := first.Search(q, 10), g.Search(q, 10)
			require.Equal(t, len(a), len(b))
			for i := range a {
				require.Equal(t, a[i].Key, b[i].Key, "round %d query %d rank %d", round, qi, i)
			}
		}
	}
}

// A graph over a few hundred vectors must find the true nearest neighbours
// almost always. Uniform Gaussian vectors are a hard case (all distances are
// close), so a larger EfSearch than the default is used here.
func TestGraph_Recall(t *testing.T) {
	const n, dim, k = 500, 64, 5
	vecs := randomVectors(n, dim, 3)
	keys := make([]string, n)
	for i := range keys {
		keys[i] = fmt.Sprintf("k%04d", i)
	}
	g := NewGraph[string]()
	g.EfSearch = 100
	for i := range keys {
		g.Add(MakeNode(keys[i], vecs[i]))
	}
	hit, total := 0, 0
	for _, q := range randomVectors(100, dim, 4) {
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
	recall := float64(hit) / float64(total)
	require.GreaterOrEqual(t, recall, 0.99, "recall@%d = %.3f", k, recall)

	// Every stored vector must find itself.
	for i := range keys {
		res := g.Search(vecs[i], 1)
		require.NotEmpty(t, res)
		require.Equal(t, keys[i], res[0].Key)
	}
}

func TestGraph_EntryIsSmallestKey(t *testing.T) {
	l := &layer[string]{nodes: map[string]*layerNode[string]{}}
	for _, k := range []string{"m", "b", "z", "a", "q"} {
		l.nodes[k] = &layerNode[string]{Node: Node[string]{Key: k}}
	}
	for i := 0; i < 20; i++ {
		require.Equal(t, "a", l.entry().Key)
	}
}
