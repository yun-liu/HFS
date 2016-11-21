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

#ifndef DISJOINT_SET
#define DISJOINT_SET

#include <vector>

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
  int rank;
  int p;
  int size;
  int size1;
} uni_elt;

class universe {
public:
	universe(int elements){
		elts = new uni_elt[elements];
		num = elements;
		for (int i = 0; i < elements; i++) {
			elts[i].rank = 0;
			elts[i].size = 1;
			elts[i].size1 = 0;
			elts[i].p = i;
		}
	}

	universe(int elements, std::vector<int> size){
		elts = new uni_elt[elements];
		num = elements;
		for (int i = 0; i < elements; i++) {
			elts[i].rank = 0;
			elts[i].size = 1;
			elts[i].size1 = size[i];
			elts[i].p = i;
		}
	}

	~universe(){
		delete[] elts;
	}

	int find(int x){
		int y = x;
		while (y != elts[y].p)
			y = elts[y].p;
		elts[x].p = y;
		return y;
	}

	void join(int x, int y){
		if (elts[x].rank > elts[y].rank) {
			elts[y].p = x;
			elts[x].size += elts[y].size;
			elts[x].size1 += elts[y].size1;
		}
		else {
			elts[x].p = y;
			elts[y].size += elts[x].size;
			elts[y].size1 += elts[x].size1;
			if (elts[x].rank == elts[y].rank)
				elts[y].rank++;
		}
		num--;
	}

    int size(int x) const { return elts[x].size; }
    int num_sets() const { return num; }

    int size1(int x) const { return elts[x].size1; }

private:
	uni_elt *elts;
    int num;
};

#endif
