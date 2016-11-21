
#include "../Stdafx.h"
#include "segment-image.h"


bool operator<(const edge &a, const edge &b)
{
	return a.w < b.w;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b, int x1, int y1, int x2, int y2)
{
	return sqrt(square(imRef(r, x1, y1) - imRef(r, x2, y2)) +
		square(imRef(g, x1, y1) - imRef(g, x2, y2)) +
		square(imRef(b, x1, y1) - imRef(b, x2, y2)));
}

/*
* Segment a graph
*
* Returns a disjoint-set forest representing the segmentation.
*
* num_vertices: number of vertices in graph.
* num_edges: number of edges in graph
* edges: array of edges.
* c: constant for treshold function.
*/
universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c) 
{
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	universe *u = new universe(num_vertices);

	// init thresholds
	float *threshold = new float[num_vertices];
	for (int i = 0; i < num_vertices; i++)
		threshold[i] = THRESHOLD(1, c);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b) {
			if ((pedge->w <= threshold[a]) &&
				(pedge->w <= threshold[b])) {
				u->join(a, b);
				a = u->find(a);
				threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
			}
		}
	}

	// free up
	delete threshold;
	return u;
}

universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c, vector<int> size)
{
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	universe *u = new universe(num_vertices, size);

	// init thresholds
	float *threshold = new float[num_vertices];
	for (int i = 0; i < num_vertices; i++)
		threshold[i] = THRESHOLD(1, c);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b) {
			if ((pedge->w <= threshold[a]) &&
				(pedge->w <= threshold[b])) {
				u->join(a, b);
				a = u->find(a);
				threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
			}
		}
	}

	// free up
	delete threshold;
	return u;
}

// Efficient Graph-Based Image Segmentation 
void segment_image(Mat &segment, Mat &img3u, float sigma, float c, int min_size)
{
	const int width = img3u.cols;
	const int height = img3u.rows;
	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);

	// smooth each color channel
	for (int y = 0; y < height; y++) {
		Vec3b *data = img3u.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++) {
			imRef(r, x, y) = data[x][0];
			imRef(g, x, y) = data[x][1];
			imRef(b, x, y) = data[x][2];
		}
	}
	image<float> *smooth_r = smooth(r, sigma);
	image<float> *smooth_g = smooth(g, sigma);
	image<float> *smooth_b = smooth(b, sigma);
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width*height * 4];
	int num = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x < width - 1) {
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x + 1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y);
				num++;
			}

			if (y < height - 1) {
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width + x;
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y + 1);
				num++;
			}

			if ((x < width - 1) && (y < height - 1)) {
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width + (x + 1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1);
				num++;
			}

			if ((x < width - 1) && (y > 0)) {
				edges[num].a = y * width + x;
				edges[num].b = (y - 1) * width + (x + 1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1);
				num++;
			}
		}
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;

	// segment
	universe *u = segment_graph(width*height, num, edges, c);

	// post process small components
	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete[] edges;
	int num_css = u->num_sets();

	// pick random colors for each component
	int *ind = new int[width*height];
	std::memset(ind, 0, width*height*sizeof(int));
	int idx = 1;
	for (int i = 0; i < height*width; i++) {
		int comp = u->find(i);
		if (!ind[comp])
			ind[comp] = idx++;
	}

	// Get the index map of segmentation
	segment.create(img3u.size(), CV_16U);
	for (int row = 0; row < segment.rows; row++) {
		ushort *data = segment.ptr<ushort>(row);
		for (int col = 0; col < segment.cols; col++) {
			int comp = u->find(row * width + col);
			data[col] = ind[comp];
		}
	}
	delete[] ind;
	delete u;
}



