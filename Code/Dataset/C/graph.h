#ifndef HIDIM_GRAPH_H
#define HIDIM_GRAPH_H

#define HIDIM_GRAPH_TYPE 0
#define HIDIM_KNN_TYPE 1

#include <string>
#include <vector>
#include <hidim/types.h>
#include <hidim/common.h>

typedef dim_t vertex_id;
typedef double edge_weight;

typedef struct node_elt_c  // shoud this be a s_edge_c structure and then a forward_list of s_edge_c? Instead of full reimplementation of the linked list
{
    struct s_node_c *node;
    edge_weight weight=1;
    struct node_elt_c *next;
} node_elt_c;

typedef struct s_node_c
{
    vertex_id id;
    dim_t outDegree=0;
    dim_t inDegree=0;
    node_elt_c *node_star = NULL;
} node_c;

namespace hidim
{
class Graph
{
public:
    Graph();
    virtual ~Graph();

    void setName(std::string name);
    void setType(dim_t type);
    dim_t getType();
    node_c V(vertex_id i);

    void addNode(vertex_id i);
    void addAllNodes(dim_t n);
    void resize(dim_t N);
    bool checkNode(vertex_id id);
    void addNeighbor(vertex_id from, vertex_id to, edge_weight weight = 1.,bool sort = false);
    void addEdge(vertex_id from, vertex_id to, edge_weight weight = 1.,bool sort = false);
    std::vector<vertex_id> getStar(vertex_id i);
    std::vector<node_elt_c> getWStar(vertex_id i);
    std::vector<vertex_id> findLeaf();

    void reverseInto(Graph *dest);
    Graph reverse();

    bool hasNeighbor(vertex_id from, vertex_id to);
    void clear();
    bool check();
    void info();

    void writeHeader(std::ofstream *stream);
    void writeData(std::ofstream *stream);
    void write(std::ofstream *stream);
    void readHeader(std::ifstream *stream);
    void readData(std::ifstream *stream);
    bool read(std::ifstream *stream);

    std::string name = "noName";
    dim_t N = 0;
    dim_t M = 0;
    bool checkSimple = false;
    bool hasSelfloop = false;
    bool isDirected = true;
    dim_t type = HIDIM_GRAPH_TYPE;

protected:
    std::vector<node_c> _V;

private:
};
}

#endif // HIDIM_GRAPH_H

