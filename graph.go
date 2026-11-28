package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"

	"github.com/coder/hnsw/heap"
	"golang.org/x/exp/maps"
)

type Vector = []float32

// Node is a node in the graph.
type Node[K cmp.Ordered] struct {
	Key   K
	Value Vector
}

func MakeNode[K cmp.Ordered](key K, vec Vector) Node[K] {
	return Node[K]{Key: key, Value: vec}
}

// layerNode is a node in a layer of the graph.
type layerNode[K cmp.Ordered] struct {
	Node[K]

	// neighbors is map of neighbor keys to neighbor nodes.
	// It is a map and not a slice to allow for efficient deletes, esp.
	// when M is high.
	neighbors map[K]*layerNode[K]
}

// addNeighbor adds a neighbor to the node. When the neighbor set overflows,
// one neighbor is dropped: with the heuristic (default) the set is re-selected
// with selectNeighbors, otherwise the neighbor with the worst distance goes.
func (n *layerNode[K]) addNeighbor(newNode *layerNode[K], m int, dist DistanceFunc, simple bool) {
	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K], m)
	}

	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}

	var drop *layerNode[K]
	if simple {
		drop = n.worstNeighbor(dist)
	} else {
		drop = n.heuristicDrop(m, dist)
	}

	delete(n.neighbors, drop.Key)
	// Delete backlink from the dropped neighbor.
	delete(drop.neighbors, n.Key)
	drop.replenish(m, dist, simple)
}

// worstNeighbor returns the neighbor farthest from n.
func (n *layerNode[K]) worstNeighbor(dist DistanceFunc) *layerNode[K] {
	var (
		worstDist = float32(math.Inf(-1))
		worst     *layerNode[K]
	)
	for _, neighbor := range n.neighbors {
		d := dist(neighbor.Value, n.Value)
		// d > worstDist may always be false if the distance function
		// returns NaN, e.g., when the embeddings are zero.
		// Ties are broken by the larger key so that the result does not
		// depend on map iteration order.
		if d > worstDist || worst == nil || (d == worstDist && neighbor.Key > worst.Key) {
			worstDist = d
			worst = neighbor
		}
	}
	return worst
}

// heuristicDrop picks the neighbor to drop from an over-full neighbor set
// (len == m+1) by running selectNeighbors over it and returning whichever
// neighbor was not selected.
func (n *layerNode[K]) heuristicDrop(m int, dist DistanceFunc) *layerNode[K] {
	cands := make([]searchCandidate[K], 0, len(n.neighbors))
	for _, nb := range n.neighbors {
		cands = append(cands, searchCandidate[K]{node: nb, dist: dist(nb.Value, n.Value)})
	}
	sortCandidates(cands)
	kept := make(map[K]struct{}, m)
	for _, c := range selectNeighbors(cands, m, dist) {
		kept[c.node.Key] = struct{}{}
	}
	for i := len(cands) - 1; i >= 0; i-- {
		if _, ok := kept[cands[i].node.Key]; !ok {
			return cands[i].node
		}
	}
	// Unreachable when len(cands) > m; fall back to the farthest neighbor.
	return n.worstNeighbor(dist)
}

// sortCandidates orders candidates by ascending distance, ties by key.
func sortCandidates[K cmp.Ordered](c []searchCandidate[K]) {
	slices.SortFunc(c, func(a, b searchCandidate[K]) int {
		if a.dist < b.dist {
			return -1
		}
		if a.dist > b.dist {
			return 1
		}
		return cmp.Compare(a.node.Key, b.node.Key)
	})
}

// selectNeighbors implements the neighbor-selection heuristic of the HNSW
// paper (Algorithm 4, with keepPrunedConnections). cands must be sorted by
// ascending distance to the base point. A candidate is kept only if it is
// closer to the base point than to every neighbor already kept; this spreads
// the edges across directions (and clusters) instead of piling them onto the
// nearest cluster. Discarded candidates then top the set up to m in distance
// order, so the node still ends up with m edges.
func selectNeighbors[K cmp.Ordered](cands []searchCandidate[K], m int, dist DistanceFunc) []searchCandidate[K] {
	if len(cands) <= m {
		return cands
	}
	selected := make([]searchCandidate[K], 0, m)
	var pruned []searchCandidate[K]
	for _, c := range cands {
		if len(selected) >= m {
			break
		}
		good := true
		for _, r := range selected {
			if dist(c.node.Value, r.node.Value) < c.dist {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, c)
		} else {
			pruned = append(pruned, c)
		}
	}
	for _, p := range pruned {
		if len(selected) >= m {
			break
		}
		selected = append(selected, p)
	}
	return selected
}

type searchCandidate[K cmp.Ordered] struct {
	node *layerNode[K]
	dist float32
}

func (s searchCandidate[K]) Less(o searchCandidate[K]) bool {
	return s.dist < o.dist
}

// search returns the nearest neighbors of target within this layer.
// Implements HNSW Algorithm 5 (SEARCH-LAYER): the result set is bounded
// by efSearch during exploration; only the k closest are returned.
func (n *layerNode[K]) search(
	// k is the number of nearest neighbors to return.
	k int,
	efSearch int,
	target Vector,
	distance DistanceFunc,
) []searchCandidate[K] {
	// The exploration budget must be at least k, otherwise the ef-bounded
	// result set cannot hold all requested neighbors and search would
	// return fewer than k results.
	if efSearch < k {
		efSearch = k
	}

	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, efSearch))
	candidates.Push(
		searchCandidate[K]{
			node: n,
			dist: distance(n.Value, target),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[K]]{}
		visited = make(map[K]bool)
	)
	result.Init(make([]searchCandidate[K], 0, efSearch))

	// Begin with the entry node in the result set.
	result.Push(candidates.Min())
	visited[n.Key] = true

	for candidates.Len() > 0 {
		current := candidates.Pop()

		// Standard HNSW termination: if the closest remaining candidate
		// is farther than the worst in the ef-bounded result set,
		// no further improvement is possible.
		if result.Len() >= efSearch && current.dist > result.Max().dist {
			break
		}

		// We iterate the map in a sorted, deterministic fashion for
		// tests.
		neighborKeys := maps.Keys(current.node.neighbors)
		slices.Sort(neighborKeys)
		for _, neighborID := range neighborKeys {
			neighbor := current.node.neighbors[neighborID]
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)
			sc := searchCandidate[K]{node: neighbor, dist: dist}
			if result.Len() < efSearch {
				result.Push(sc)
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(sc)
			} else {
				continue
			}
			candidates.Push(sc)
		}
	}

	// Return only the k closest from the ef-sized result set.
	res := result.Slice()
	slices.SortFunc(res, func(a, b searchCandidate[K]) int {
		if a.dist < b.dist {
			return -1
		}
		if a.dist > b.dist {
			return 1
		}
		// Deterministic tiebreaking by key.
		return cmp.Compare(a.node.Key, b.node.Key)
	})
	if len(res) > k {
		res = res[:k]
	}
	return res
}

func (n *layerNode[K]) replenish(m int, dist DistanceFunc, simple bool) {
	if len(n.neighbors) >= m {
		return
	}

	// Restore connectivity by adding new neighbors.
	// This is a naive implementation that could be improved by
	// using a priority queue to find the best candidates.
	//
	// Both loops iterate in sorted key order so the result is deterministic.
	for _, nk := range sortedKeys(n.neighbors) {
		neighbor := n.neighbors[nk]
		for _, key := range sortedKeys(neighbor.neighbors) {
			candidate := neighbor.neighbors[key]
			if _, ok := n.neighbors[key]; ok {
				// do not add duplicates
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, dist, simple)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

// sortedKeys returns the keys of m in ascending order.
func sortedKeys[K cmp.Ordered](m map[K]*layerNode[K]) []K {
	keys := maps.Keys(m)
	slices.Sort(keys)
	return keys
}

// isolates remove the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[K]) isolate(m int, dist DistanceFunc, simple bool) {
	for _, neighbor := range n.neighbors {
		delete(neighbor.neighbors, n.Key)
	}

	for _, k := range sortedKeys(n.neighbors) {
		n.neighbors[k].replenish(m, dist, simple)
	}
}

type layer[K cmp.Ordered] struct {
	// nodes is a map of nodes IDs to nodes.
	// All nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	//
	// nodes is exported for interop with encoding/gob.
	nodes map[K]*layerNode[K]
}

// entry returns the entry node of the layer.
// It is the node with the smallest key, so that the same set of nodes
// always yields the same entry point regardless of map iteration order.
// The scan is O(size), which is cheap for the upper layers where it
// matters; use anyNode when any node will do.
func (l *layer[K]) entry() *layerNode[K] {
	if l == nil {
		return nil
	}
	var best *layerNode[K]
	for k, node := range l.nodes {
		if best == nil || k < best.Key {
			best = node
		}
	}
	return best
}

// anyNode returns an arbitrary node of the layer in O(1).
func (l *layer[K]) anyNode() *layerNode[K] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		return node
	}
	return nil
}

func (l *layer[K]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// All public parameters must be set before adding nodes to the graph.
// K is cmp.Ordered instead of of comparable so that they can be sorted.
type Graph[K cmp.Ordered] struct {
	// Distance is the distance function used to compare embeddings.
	Distance DistanceFunc

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// M is the maximum number of neighbors to keep for each node.
	// A good default for OpenAI embeddings is 16.
	M int

	// Ml is the level generation factor.
	// E.g., for Ml = 0.25, each layer is 1/4 the size of the previous layer.
	Ml float64

	// SimpleNeighbors selects the plain "keep the M nearest" neighbor policy
	// instead of the paper's heuristic (Algorithm 4). The heuristic is the
	// default: it keeps well-connected graphs on clustered data. The simple
	// policy is cheaper to build.
	SimpleNeighbors bool

	// EfSearch is the number of nodes to consider in the search phase.
	// 20 is a reasonable default. Higher values improve search accuracy at
	// the expense of memory.
	EfSearch int

	// layers is a slice of layers in the graph.
	layers []*layer[K]
}

// defaultSeed is the fixed seed used by NewGraph so that building a graph
// from the same input sequence yields the same graph. Callers that want
// randomised levels can set Graph.Rng themselves.
const defaultSeed = 1

func defaultRand() *rand.Rand {
	return rand.New(rand.NewSource(defaultSeed))
}

// NewGraph returns a new graph with default parameters, roughly designed for
// storing OpenAI embeddings.
func NewGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 20,
		Rng:      defaultRand(),
	}
}

// maxLevel returns an upper-bound on the number of levels in the graph
// based on the size of the base layer.
func maxLevel(ml float64, numNodes int) int {
	if ml == 0 {
		panic("ml must be greater than 0")
	}

	if numNodes == 0 {
		return 1
	}

	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)

	m := int(math.Round(l)) + 1

	return m
}

// randomLevel generates a random level for a new node.
func (h *Graph[K]) randomLevel() int {
	// max avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	max := 1
	if len(h.layers) > 0 {
		if h.Ml == 0 {
			panic("(*Graph).Ml must be greater than 0")
		}
		max = maxLevel(h.Ml, h.layers[0].size())
	}

	for level := 0; level < max; level++ {
		if h.Rng == nil {
			h.Rng = defaultRand()
		}
		r := h.Rng.Float64()
		if r > h.Ml {
			return level
		}
	}

	return max
}

func (g *Graph[K]) assertDims(n Vector) {
	if len(g.layers) == 0 {
		return
	}
	hasDims := g.Dims()
	if hasDims != len(n) {
		panic(fmt.Sprint("embedding dimension mismatch: ", hasDims, " != ", len(n)))
	}
}

// Dims returns the number of dimensions in the graph, or
// 0 if the graph is empty.
func (g *Graph[K]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	return len(g.layers[0].anyNode().Value)
}

func ptr[T any](v T) *T {
	return &v
}

// replaceEfFactor scales the search width used when Add replaces an existing
// key (an in-place update). With a plain EfSearch-wide search, recall after
// repeated updates falls well below a graph built from scratch from the same
// data; a 4x wider search on the update path recovers most of that gap while
// leaving first-time inserts at their usual cost.
const replaceEfFactor = 4

// Add inserts nodes into the graph.
// If another node with the same ID exists, it is replaced.
func (g *Graph[K]) Add(nodes ...Node[K]) {
	for _, node := range nodes {
		key := node.Key
		vec := node.Value

		g.assertDims(vec)

		// Adding an existing key replaces it. Remove the old node from every
		// layer up front: removing it layer by layer while inserting would
		// delete the lower layers' copies too, leaving the elevator pointing
		// at a node that is no longer there. Delete may also drop layers that
		// become empty, so it has to run before the layers for this insert
		// are created.
		replaced := g.Delete(key)

		insertLevel := g.randomLevel()
		// Create layers that don't exist yet.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K]{})
		}

		if insertLevel < 0 {
			panic("invalid level")
		}

		var elevator *K

		preLen := g.Len()

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[K]{
				Node: Node[K]{
					Key:   key,
					Value: vec,
				},
			}

			// Insert the new node into the layer.
			if layer.anyNode() == nil {
				layer.nodes = map[K]*layerNode[K]{key: newNode}
				continue
			}

			// Now at the highest layer with more than one node, so we can begin
			// searching for the best way to enter the graph.
			// On subsequent layers, we use the elevator node to enter the graph
			// at the best point. Only the topmost layer needs entry(), and
			// that layer is small, so the O(size) scan is cheap.
			var searchPoint *layerNode[K]
			if elevator != nil {
				searchPoint = layer.nodes[*elevator]
			} else {
				searchPoint = layer.entry()
			}

			if g.Distance == nil {
				panic("(*Graph).Distance must be set")
			}

			// Candidate pool: the heuristic needs more than M candidates to
			// choose from; the simple policy only ever uses the M nearest.
			k := g.M
			if !g.SimpleNeighbors && g.EfSearch > k {
				k = g.EfSearch
			}
			efBuild := g.EfSearch
			if replaced && !g.SimpleNeighbors {
				// An updated node is re-inserted into a graph that has
				// already been shaped around its old position, and the
				// graph does not repair itself when nodes move, so a
				// wider search is needed to find good neighbors for it.
				// See replaceEfFactor.
				efBuild = g.EfSearch * replaceEfFactor
				if efBuild > k {
					k = efBuild
				}
			}
			neighborhood := searchPoint.search(k, efBuild, vec, g.Distance)
			if len(neighborhood) == 0 {
				// This should never happen because the searchPoint itself
				// should be in the result set.
				panic("no nodes found")
			}

			// Re-set the elevator node for the next layer.
			elevator = ptr(neighborhood[0].node.Key)

			if g.SimpleNeighbors {
				if len(neighborhood) > g.M {
					neighborhood = neighborhood[:g.M]
				}
			} else {
				neighborhood = selectNeighbors(neighborhood, g.M, g.Distance)
			}

			if insertLevel >= i {
				// Insert the new node into the layer.
				layer.nodes[key] = newNode
				for _, node := range neighborhood {
					// Create a bi-directional edge between the new node and the best node.
					node.node.addNeighbor(newNode, g.M, g.Distance, g.SimpleNeighbors)
					newNode.addNeighbor(node.node, g.M, g.Distance, g.SimpleNeighbors)
				}
			}
		}

		// Invariant check: the node should have been added to the graph.
		if g.Len() != preLen+1 {
			if len(g.layers) > 0 && g.layers[len(g.layers)-1].anyNode() == nil {
				g.layers = g.layers[:len(g.layers)-1]
			}
		}
	}
}

// Search finds the k nearest neighbors from the target node.
func (h *Graph[K]) Search(near Vector, k int) []Node[K] {
	sr := h.search(near, k)
	out := make([]Node[K], len(sr))
	for i, node := range sr {
		out[i] = node.Node
	}
	return out
}

// SearchWithDistance finds the k nearest neighbors from the target node
// and returns the distance.
func (h *Graph[K]) SearchWithDistance(near Vector, k int) []SearchResult[K] {
	return h.search(near, k)
}

type SearchResult[T cmp.Ordered] struct {
	Node[T]
	Distance float32
}

func (h *Graph[K]) search(near Vector, k int) []SearchResult[K] {
	h.assertDims(near)
	if len(h.layers) == 0 {
		return nil
	}

	var (
		efSearch = h.EfSearch

		elevator *K
	)

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		var searchPoint *layerNode[K]
		if elevator != nil {
			searchPoint = h.layers[layer].nodes[*elevator]
		} else {
			searchPoint = h.layers[layer].entry()
		}

		// Descending hierarchies
		if layer > 0 {
			nodes := searchPoint.search(1, efSearch, near, h.Distance)
			elevator = ptr(nodes[0].node.Key)
			continue
		}

		nodes := searchPoint.search(k, efSearch, near, h.Distance)
		out := make([]SearchResult[K], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, SearchResult[K]{
				Node:     node.node.Node,
				Distance: node.dist,
			})
		}

		return out
	}

	panic("unreachable")
}

// Len returns the number of nodes in the graph.
func (h *Graph[K]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph by key.
// It tries to preserve the clustering properties of the graph by
// replenishing connectivity in the affected neighborhoods.
func (h *Graph[K]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false
	}

	deleteLayer := map[int]struct{}{}
	var deleted bool
	for i, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue
		}
		delete(layer.nodes, key)
		if len(layer.nodes) == 0 {
			deleteLayer[i] = struct{}{}
		}
		// Edges are not guaranteed to be symmetric (replenish and the
		// over-capacity drop only touch one side), so other nodes may still
		// point at the deleted node even though it does not point back.
		// Purge those before replenishing, or searches could walk into a
		// node that is no longer in the graph.
		for _, other := range layer.nodes {
			delete(other.neighbors, key)
		}
		node.isolate(h.M, h.Distance, h.SimpleNeighbors)
		deleted = true
	}

	if len(deleteLayer) > 0 {
		newLayers := make([]*layer[K], 0, len(h.layers)-len(deleteLayer))
		for i, layer := range h.layers {
			if _, ok := deleteLayer[i]; ok {
				continue
			}
			newLayers = append(newLayers, layer)
		}

		h.layers = newLayers
	}

	return deleted
}

// Lookup returns the vector with the given key.
func (h *Graph[K]) Lookup(key K) (Vector, bool) {
	if len(h.layers) == 0 {
		return nil, false
	}

	node, ok := h.layers[0].nodes[key]
	if !ok {
		return nil, false
	}
	return node.Value, ok
}
