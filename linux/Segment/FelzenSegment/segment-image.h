/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include "image.h"
#include "filter.h"
#include "disjoint-set.h"


// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct
{
	float w;
	int a, b;
} edge;

universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c);
universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c, vector<int> size);

void segment_image(Mat &segment, Mat &img3u, float sigma, float c, int min_size);


#endif
